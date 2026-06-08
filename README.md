Entity-Guided Reinforcement Learning for Factual Legal Summarization”

Overview

This repository contains:

* TANHA entity alignment framework
* Entity-guided reward formulation
* Reinforcement learning training scripts using GRPO
* Ablation variants of TANHA
* Legal NER annotation interface
* Human evaluation interface

TANHA aligns legal entities between a source judgment and a generated summary using a combination of:

* Canonicalization
* Lexical matching
* Semantic matching

The resulting alignments are used to compute entity-level coverage and hallucination signals, which serve as rewards during reinforcement learning.

⸻

Repository Structure

.
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── train_entity_cache.pkl
│   └── val_entity_cache.pkl
│
├── tanha/
│   ├── tanha_full.py
│   ├── tanha_no_canonicalization.py
│   ├── tanha_no_lexical.py
│   └── tanha_no_semantic.py
│
├── rl/
│   ├── train_rl_full.py
│   ├── train_rl_no_canonicalization.py
│   ├── train_rl_no_lexical.py
│   └── train_rl_no_semantic.py
│
├── annotation_tools/
│   ├── ner_annotation.html
│   └── human_evaluation.html
│
├── requirements.txt
└── README.md

⸻

Installation

Create a fresh environment:

conda create -n tanha python=3.11
conda activate tanha

Install dependencies:

pip install -r requirements.txt

⸻

Required Models

Sentence Embeddings

TANHA uses:

BAAI/bge-large-en-v1.5

for semantic entity matching.

Legal NER Model

The reward pipeline expects a legal-domain NER model:

en_legal_ner_trf

Install or place the model in your local environment before training.

⸻

Data Format

Training and validation files should be JSONL files containing at least:

{
  "EN_Judgment": "...",
  "SANITIZED_SUMMARY": "..."
}

⸻

Entity Cache

The RL reward uses pre-computed entity caches.

Each cache contains:

{
    "source_metadata": ...,
    "source_embeddings": ...,
    "gold_metadata": ...,
    "gold_embeddings": ...
}

These caches are generated before RL training to avoid repeated NER extraction and embedding computation.

⸻

Running TANHA Alignment

Example:

python tanha/tanha_full.py

Ablation variants:

python tanha/tanha_no_canonicalization.py
python tanha/tanha_no_lexical.py
python tanha/tanha_no_semantic.py

⸻

Running Reinforcement Learning

Full TANHA reward:

python rl/train_rl_full.py

No Canonicalization:

python rl/train_rl_no_canonicalization.py

No Lexical Matching:

python rl/train_rl_no_lexical.py

No Semantic Matching:

python rl/train_rl_no_semantic.py

⸻

Reward Formulation

The RL reward is computed from TANHA alignments.

Coverage:

Coverage =
\sum_{t \in \mathcal{T}}
w_t
\frac{|E_{match,t}|}
{|E_{source,t}|}

Hallucination:

Hallucination =
\sum_{t \in \mathcal{T}}
w_t
\frac{|E_{unmatched,t}|}
{|E_{generated,t}|}

Final reward:

R =
\alpha \cdot Coverage
-
\beta \cdot Hallucination

where the entity-type weights are derived from dataset-level retention statistics.

⸻

Human Evaluation

The repository contains browser-based interfaces for:

Named Entity Annotation

annotation_tools/ner_annotation.html

Human Evaluation

annotation_tools/human_evaluation.html

These interfaces were used during dataset construction and evaluation.

⸻

Reproducibility

Experiments were conducted using:

* Fixed random seed
* LoRA fine-tuning
* GRPO optimization
* Entity-guided reward derived from TANHA

Hyperparameters used in the paper are provided in the corresponding training scripts.

⸻

Citation

If you use this repository, please cite:

@inproceedings{anonymous2026tanha,
  title={TANHA: Entity-Guided Reinforcement Learning for Factual Legal Summarization},
  author={Anonymous},
  year={2026}
}

The citation will be updated after publication.

A couple of notes:

1. Replace Anonymous only after the review process is complete.
2. If you’re submitting double-blind, don’t include trained checkpoints, WandB links, institution names, usernames, server paths, or dataset download links that reveal authorship.
