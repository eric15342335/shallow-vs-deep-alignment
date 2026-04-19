from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen3 has no default system prompt per official documentation
default_system_prompt = ""


def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    # transformers>=5.x dropped use_cache as a model constructor arg
    model_kwargs = {k: v for k, v in model_kwargs.items() if k != "use_cache"}
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    # Qwen tokenizer already has pad_token_id=151643 (<|endoftext|>) set in generation_config;
    # add_bos_token and add_eos_token are both false in the tokenizer config.
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
    """Convert OpenAI chat format to Qwen3 ChatML format for inference."""

    def string_formatter_completion_only(example):
        # ChatML format: <|im_start|>role\ncontent<|im_end|>\n
        # Last assistant message has no <|im_end|> — model generates the rest.
        IM_START = "<|im_start|>"
        IM_END = "<|im_end|>"

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

        str_message = ""
        if system_prompt:
            str_message = IM_START + "system\n" + system_prompt + IM_END + "\n"

        while pt < len(messages) - 1:
            if messages[pt]["role"] != "user":
                raise ValueError("the message should be user - assistant alternation")
            str_message += IM_START + "user\n" + messages[pt]["content"] + IM_END + "\n"
            pt += 1
            if pt >= len(messages) - 1:
                break
            if messages[pt]["role"] != "assistant":
                raise ValueError("the message should be user - assistant alternation")
            str_message += IM_START + "assistant\n" + messages[pt]["content"] + IM_END + "\n"
            pt += 1

        if messages[-1]["role"] != "assistant":
            raise ValueError("completion only mode should end with a header of assistant message")

        str_message += IM_START + "assistant\n" + messages[-1]["content"]
        return {"text": str_message}
