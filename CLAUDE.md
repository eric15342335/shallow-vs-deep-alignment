# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a single arXiv paper archive (`2406.05946`) — a NeurIPS 2024 research paper titled **"Safety Alignment Should Be Made More Than Just a Few Tokens Deep"** by researchers from Princeton University and Google DeepMind.

## Working with the Archive

The file `2406.05946` is a gzip-compressed tar archive. To extract and compile:

```bash
# Extract
gzip -cd 2406.05946 | tar -x

# Compile (requires a LaTeX distribution with NeurIPS 2024 packages)
pdflatex neurips_2024.tex
bibtex neurips_2024
pdflatex neurips_2024.tex
pdflatex neurips_2024.tex
```

## Paper Structure

After extraction, the layout is:

- `neurips_2024.tex` — main document, imports all sections
- `reference.bib` — BibTeX bibliography
- `sections/` — individual section files (introduction, prelim, superficial_alignment, data_augmentation, loss_function, related, conclusion, appendices/)
- `figs/` — figures and visualizations
- `*.sty` — NeurIPS 2024 style files and helper packages

## Core Research Concepts

**Problem:** Current LLM safety alignment is "shallow" — it effectively controls only the first few output tokens (refusal prefixes like "I cannot..."). Models remain vulnerable to attacks that bypass this shallow layer.

**Attack types analyzed:**
- Adversarial suffix attacks (GCG)
- Prefilling attacks (forcing harmful starting tokens)
- Decoding parameter exploits
- Fine-tuning attacks

**Defenses proposed:**
1. **Data augmentation** — train on harmful-prefix-then-safe-continuation trajectories to deepen alignment
2. **Constrained fine-tuning (SoftSFT)** — adds a penalty term (controlled by β) that constrains how much initial token probabilities can shift during fine-tuning

**Key metric:** Token-depth analysis using gradient norms, KL divergence, and cross-entropy loss to measure where safety enforcement concentrates in the output sequence.

## LaTeX Conventions

The paper uses custom comment macros for author collaboration:
```latex
\xiangyu{...}   % red
\prateek{...}   % blue
\kaifeng{...}   % orange
\ashwinee{...}  % red
\peter{...}     % red
\ahmad{...}     % red
```

Safe/unsafe dialogue examples are typeset using `tcolorbox` environments defined in the preamble.
