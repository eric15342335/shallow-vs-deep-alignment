import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

default_system_prompt = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
)


def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    # transformers>=5.x dropped use_cache as a model constructor arg
    model_kwargs = {k: v for k, v in model_kwargs.items() if k != "use_cache"}
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model.config._name_or_path, add_eos_token=False, add_bos_token=False
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side
    return model, tokenizer


class QwenStringConverter:
    """Convert OpenAI chat format to plain-text instruction format for Qwen base models."""

    def string_formatter_completion_only(example):
        # Plain-text format consistent with llama2_base / gemma_base.
        B_INST = "\n### Instruction:\n"
        E_INST = "\n### Response:\n"
        END = "\n"

        if "messages" not in example:
            raise ValueError("No messages in the example")
        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")

        pt = 0
        if messages[0]["role"] != "system":
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]["content"]
            pt = 1

        str_message = system_prompt
        first_round = True

        while pt < len(messages) - 1:
            if messages[pt]["role"] != "user":
                raise ValueError("the message should be user - assistant alternation")
            if first_round:
                str_message += B_INST + messages[pt]["content"] + END + E_INST
                first_round = False
            else:
                str_message += B_INST + messages[pt]["content"] + END + E_INST
            pt += 1
            if pt >= len(messages) - 1:
                break
            if messages[pt]["role"] != "assistant":
                raise ValueError("the message should be user - assistant alternation")
            str_message += messages[pt]["content"] + END
            pt += 1

        if messages[-1]["role"] != "assistant":
            raise ValueError("completion only mode should end with a header of assistant message")

        str_message += messages[-1]["content"]
        return {"text": str_message}


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords):
        self.stops = keywords

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        stops = self.stops.to(input_ids.device)
        decisions = []
        for seq in input_ids:
            flag = False
            for stop in stops:
                if len(seq) >= len(stop) and torch.all((stop == seq[-len(stop):])).item():
                    flag = True
                    break
            decisions.append(flag)
        return torch.tensor(decisions).to(input_ids.device)


# <|endoftext|> token ID = 151643 in Qwen3's tokenizer (confirmed from tokenizer_config.json)
# eos_token_id in generation_config is [151645, 151643], so 151643 is a valid stop token.
base_stop_key_words = torch.LongTensor([[151643]])
base_stopping_criteria = StoppingCriteriaList([KeywordStoppingCriteria(keywords=base_stop_key_words)])
