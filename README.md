# Entity-Guided Reinforcement Learning for Factual Legal Summarization

## Overview

This repository accompanies our paper on **Entity-Guided Reinforcement Learning for Factual Legal Summarization**.

The proposed framework consists of two components:

1. **TANHA (Type-Aware Normalization and Hybrid Alignment)**, which aligns legal entities between the source judgment and the generated summary using:
   - Canonicalization
   - Lexical similarity
   - Semantic similarity

2. **Entity-guided Reinforcement Learning**, where the alignments produced by TANHA are used to compute a structural reward based on:
   - Source entity coverage
   - Entity hallucination penalty
   - Length regularization

The repository also includes all ablation implementations, annotation interfaces, and human evaluation tools used in the paper.

---

# Repository Structure

```
.
├── ablations/
│   ├── tanha_full.py
│   ├── tanha_no_canonicalization.py
│   ├── tanha_no_lexical.py
│   └── tanha_no_semantic.py
│
├── rl_reward_full.py
│
├── human_validation.html
├── ner_validator.html
├── requirements.txt
└── README.md
```

---

# File Description

## Reinforcement Learning

### `rl_reward_full.py`

Implements the complete reinforcement learning framework described in the paper.

This script includes

- TANHA entity alignment
- Structural entity reward
- Source-entity coverage reward
- Hallucination penalty
- Length regularization
- GRPO training
- Validation and diagnostic utilities

Running this script reproduces the proposed RL training framework described in the paper.

---

## TANHA Ablation Studies

The `ablations/` directory contains the alignment variants used for the ablation study.

### `tanha_full.py`

Full TANHA alignment:

- Canonicalization
- Lexical matching
- Semantic matching

This corresponds to the complete alignment method proposed in the paper.

---

### `tanha_no_canonicalization.py`

Removes the canonicalization stage while keeping lexical and semantic matching unchanged.

Used for the "Without Canonicalization" ablation.

---

### `tanha_no_lexical.py`

Removes lexical matching while retaining canonicalization and semantic similarity.

Used for the "Without Lexical Matching" ablation.

---

### `tanha_no_semantic.py`

Removes semantic similarity matching while retaining canonicalization and lexical matching.

Used for the "Without Semantic Matching" ablation.

---

# Installation

Create a new environment

```bash
conda create -n tanha python=3.11
conda activate tanha
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Required Models

## Sentence Embedding Model

Semantic entity matching uses

```
BAAI/bge-large-en-v1.5
```

---

## Legal Named Entity Recognition

Entity extraction requires

```
en_legal_ner_trf
```

Install using

```bash
pip install https://huggingface.co/ali6parmak/en_legal_ner_trf/resolve/main/en_legal_ner_trf-3.2.0-py3-none-any.whl
```

---

# Dataset Format

Training and validation datasets are expected in JSONL format.

Minimum fields:

```json
{
    "EN_Judgment": "...",
    "SANITIZED_SUMMARY": "..."
}
```

---

# Entity Cache

To avoid repeated NER extraction and sentence embedding computation during RL training, pre-computed entity caches are used.

Each cache contains

```
source_metadata
source_embeddings
```

The RL implementation only requires source-side entity information.

---

# Running the Experiments

## Proposed Method

Run the complete reinforcement learning framework

```bash
python rl_reward_full.py
```

---

## TANHA Ablation Experiments

Full TANHA

```bash
python ablations/tanha_full.py
```

Without Canonicalization

```bash
python ablations/tanha_no_canonicalization.py
```

Without Lexical Matching

```bash
python ablations/tanha_no_lexical.py
```

Without Semantic Matching

```bash
python ablations/tanha_no_semantic.py
```

---

# Annotation Tools

## `ner_validator.html`

Browser-based interface used for validating legal named entity annotations.

---

## `human_validation.html`

Browser-based interface used for human evaluation of generated summaries.
