

### Entity-Guided Reinforcement Learning for Factual Legal Summarization

## Overview

This repository contains:

- TANHA entity alignment framework
- Entity-guided reward formulation
- Reinforcement learning training scripts using GRPO
- Ablation variants of TANHA
- Legal NER annotation interface
- Human evaluation interface

TANHA aligns legal entities between a source judgment and a generated summary using a combination of:

- Canonicalization
- Lexical matching
- Semantic matching

The resulting alignments are used to compute entity-level coverage and hallucination signals, which are subsequently used as rewards during reinforcement learning.

---

## Repository Structure

text . ├── data/ │   ├── train.jsonl │   ├── val.jsonl │   ├── train_entity_cache.pkl │   └── val_entity_cache.pkl │ ├── tanha/ │   ├── tanha_full.py │   ├── tanha_no_canonicalization.py │   ├── tanha_no_lexical.py │   └── tanha_no_semantic.py │ ├── rl/ │   ├── train_rl_full.py │   ├── train_rl_no_canonicalization.py │   ├── train_rl_no_lexical.py │   └── train_rl_no_semantic.py │ ├── annotation_tools/ │   ├── ner_annotation.html │   └── human_evaluation.html │ ├── requirements.txt └── README.md 

---

## Installation

Create a new environment:

bash conda create -n tanha python=3.11 conda activate tanha 

Install dependencies:

bash pip install -r requirements.txt 

---

## Required Models

### Sentence Embedding Model

TANHA uses the following sentence embedding model for semantic entity matching:

text BAAI/bge-large-en-v1.5 

### Legal NER Model

The reinforcement learning reward pipeline requires a legal-domain NER model:

text en_legal_ner_trf 

Ensure that the model is installed and accessible in the local environment before running training scripts.

pip install https://huggingface.co/ali6parmak/en_legal_ner_trf/resolve/main/en_legal_ner_trf-3.2.0-py3-none-any.whl


---

## Data Format

Training and validation datasets should be provided in JSONL format and contain at least the following fields:

json {   "EN_Judgment": "...",   "SANITIZED_SUMMARY": "..." } 

---

## Entity Cache

To avoid repeated NER extraction and embedding computation during RL training, TANHA uses pre-computed entity caches.

Each cache entry contains:

python {     "source_metadata": ...,     "source_embeddings": ...,     "gold_metadata": ...,     "gold_embeddings": ... } 

---

## Running TANHA Alignment

Run the complete TANHA alignment framework:

bash python tanha/tanha_full.py 

### Ablation Variants

Without canonicalization:

bash python tanha/tanha_no_canonicalization.py 

Without lexical matching:

bash python tanha/tanha_no_lexical.py 

Without semantic matching:

bash python tanha/tanha_no_semantic.py 

---

## Running Reinforcement Learning

Full TANHA reward:

bash python rl/train_rl_full.py 

No canonicalization:

bash python rl/train_rl_no_canonicalization.py 

No lexical matching:

bash python rl/train_rl_no_lexical.py 

No semantic matching:

bash python rl/train_rl_no_semantic.py 

---

## Reward Formulation

The reinforcement learning reward is computed directly from TANHA entity alignments.

### Coverage

[
\text{Coverage}
=
\sum_{t \in \mathcal{T}}
w_t
\frac{|E_{\text{match},t}|}
{|E_{\text{source},t}|}
]

### Hallucination

[
\text{Hallucination}
=
\sum_{t \in \mathcal{T}}
w_t
\frac{|E_{\text{unmatched},t}|}
{|E_{\text{generated},t}|}
]

### Final Reward

[
R
=
\alpha \cdot \text{Coverage}
-
\beta \cdot \text{Hallucination}
]

where the entity-type weights are derived from dataset-level retention statistics.

---

## Annotation and Human Evaluation Tools

The repository includes browser-based interfaces used during dataset creation and evaluation.

### Named Entity Annotation

text annotation_tools/ner_annotation.html 

### Human Evaluation

text annotation_tools/human_evaluation.html 

---

## Reproducibility

Experiments reported in the paper were conducted using:

- Fixed random seed
- LoRA fine-tuning
- GRPO optimization
- TANHA-based entity alignment
- Entity-guided reinforcement learning reward

All major hyperparameters are provided in the corresponding training scripts.

---
