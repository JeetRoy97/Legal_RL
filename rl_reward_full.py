# %%
import os
os.environ["VLLM_USE_V1"] = "0" # Force V0
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0" # Disable V1's compiler overhead
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from unsloth import FastLanguageModel
import torch
import math
max_seq_length = 6092 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

from legal_structural_utils_v2 import *
from debug_structural import *

import wandb
generation_config = {
    "max_new_tokens": 900,
    "min_new_tokens": 600,
    "temperature": 0.8,
    "top_p": 0.95,
}

wandb.init(
    project="MILDSum_GRPO",
    name="modified_entity_reward_v2_earlystopping",
    config=
        generation_config
)

from collections import Counter
import numpy as np

def compute_type_importance(entity_cache):

    counts = Counter()

    for sample in entity_cache.values():
        for _, t in sample["source_metadata"].keys():
            counts[t] += 1

    max_count = max(counts.values())

    weights = {}

    for t, c in counts.items():
        weights[t] = np.sqrt(max_count / c)

    return weights


device = torch.device("cuda")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    # model_name="/scratch/roy.2/MILDSum/llama_en_sft_only500/checkpoint-200",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    max_lora_rank=lora_rank,
    fast_inference = True,
    gpu_memory_utilization=0.5,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.2",
)

# %%
import pandas as pd
train_df =pd.read_json('/csehome/roy.2/MILDSum/v3/train_clean_hi_tagged.json', lines=True)
val_df =pd.read_json('/csehome/roy.2/MILDSum/v3/val_clean_hi_tagged.json', lines=True)

train_df = train_df.reset_index()   # keep old index
val_df   = val_df.reset_index()

train_df.rename(columns={"index": "orig_id"}, inplace=True)
val_df.rename(columns={"index": "orig_id"}, inplace=True)

train_df["id"] = train_df.index.astype(int)
val_df["id"]   = val_df.index.astype(int)

def get_token_length(text):
    return len(tokenizer(text, truncation=False)["input_ids"])

train_df["token_len"] = train_df["EN_Judgment"].apply(get_token_length)
val_df["token_len"] = val_df["EN_Judgment"].apply(get_token_length)

MAX_ALLOWED = 4500  # leave buffer under 16384
original_size = len(train_df)


train_df = train_df[train_df["token_len"] <= MAX_ALLOWED]

print(f"Filtered {original_size - len(train_df)} long samples")
print(f"New dataset size: {len(train_df)}")

original_size = len(val_df)

val_df   = val_df[val_df["token_len"] <= MAX_ALLOWED]
print(f"Filtered {original_size - len(val_df)} long samples")
print(f"New dataset size: {len(val_df)}")


train_df["id"] = train_df["orig_id"]
val_df["id"]   = val_df["orig_id"]


prompt="""Summarize the TARGET JUDGMENT provided by the user.
### INSTRUCTIONS
The summary should include Case Details, Background & Facts, Core Legal Issues, Key Arguments (Petitioner vs Respondent), Precedents Cited, Judicial Reasoning (Ratio Decidendi) and Final Decision

### Constraints:
- Formal legal tone
- No hallucination or invented facts
- Omit missing information
- Do not copy verbatim from the judgment, instead synthesize in your own words
- You MUST include the important legal entities: COURT, JUDGE,STATUTE, PROVISION, PRECEDENT, PETITIONER, RESPONDENT
You MUST explicitly include:

- Statutes (e.g., IPC, CrPC, etc.)
- Provisions (e.g., Section 302, Section 313, etc.)
- Precedents (case laws)

If any of these exist in the judgment and are missing in your summary, the answer is considered incorrect.
"""

import re


train_df["base_prompt"] = prompt
val_df["base_prompt"] = prompt

train_df['input'] = (
    "\n[SAMPLE_ID=" + train_df["id"].astype(str) + "]\n" +
    train_df['base_prompt'] +
    "\nTARGET JUDGMENT:\n" +
    train_df['EN_Judgment'] +
    "\nSUMMARY:\n"
)        
train_df['output'] = train_df['SANITIZED_SUMMARY']



val_df['input'] = (
    "\n[SAMPLE_ID=" + val_df["id"].astype(str) + "]\n" +
    val_df['base_prompt'] +
    "\nTARGET JUDGMENT:\n" +
    val_df['EN_Judgment'] +
    "\nSUMMARY:\n"
)
val_df['output'] = val_df['SANITIZED_SUMMARY']


from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df[["id", "input"]])
val_dataset = Dataset.from_pandas(val_df[["id", "input"]])

# %%
def create_conversation(row):
    return [
        {"from": "human", "value": row["input"]},
    ]

train_df["conversations"] = train_df.apply(create_conversation, axis=1)
val_df["conversations"] = val_df.apply(create_conversation, axis=1)

from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df[["id", "conversations"]])
val_dataset = Dataset.from_pandas(val_df[["id", "conversations"]])

from unsloth.chat_templates import standardize_sharegpt

train_dataset = standardize_sharegpt(train_dataset)
val_dataset = standardize_sharegpt(val_dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True   # IMPORTANT for RL
        )
        for convo in convos
    ]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

# %%
train_dataset = train_dataset.rename_column("text", "prompt")
val_dataset = val_dataset.rename_column("text", "prompt")

train_dataset = train_dataset.remove_columns(["conversations"])
val_dataset = val_dataset.remove_columns(["conversations"])

# %%
val_dataset[0]

# %%
import pickle

with open("/csehome/roy.2/MILDSum/v3/train_entity_cache_structural.pkl", "rb") as f:
    train_entity_cache = pickle.load(f)

with open("/csehome/roy.2/MILDSum/v3/val_entity_cache_structural.pkl", "rb") as f:
    val_entity_cache = pickle.load(f)

# If it's list, convert to dict
if isinstance(train_entity_cache, list):
    train_entity_cache = {i: sample for i, sample in enumerate(train_entity_cache)}

if isinstance(val_entity_cache, list):
    val_entity_cache = {i: sample for i, sample in enumerate(val_entity_cache)}


def filter_low_entity_samples(dataset, entity_cache, min_entities=5):

    keep_indices = []

    for i, sample in enumerate(dataset):
        sid = sample["id"]
        cache = entity_cache[sid]
        num_source = len(cache["source_metadata"])

        if num_source >= min_entities:
            keep_indices.append(i)

    return dataset.select(keep_indices)

print("Original train size:", len(train_dataset))
train_dataset = filter_low_entity_samples(
    train_dataset,
    train_entity_cache,
    min_entities=5
)
print("Filtered train size:", len(train_dataset))
print("Original val size:", len(val_dataset))
val_dataset = filter_low_entity_samples(
    val_dataset,
    val_entity_cache,
    min_entities=5
)
print("Filtered val size:", len(val_dataset))


# %%
import spacy

legal_nlp = spacy.load("en_legal_ner_trf")
KEEP_LABELS = {
    "COURT",
    "STATUTE",
    "PROVISION",
    "PRECEDENT",
    "PETITIONER",
    "RESPONDENT",
    "JUDGE",
}

FIRST_ONLY_LABELS = {
    "COURT",
    "STATUTE",
    "PRECEDENT",
}

def filter_legal_entities(doc):
    filtered = []
    seen = set()

    for ent in sorted(doc.ents, key=lambda e: e.start_char):

        if ent.label_ not in KEEP_LABELS:
            continue

        key = (ent.text.lower().strip(), ent.label_)

        if ent.label_ in FIRST_ONLY_LABELS:
            if key in seen:
                continue
            seen.add(key)

        filtered.append(ent)

    return filtered

def extract_generated_entities_spacy(text):
    """
    Extract standardized (text, label) entity tuples
    from a single generated summary.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    try:
        doc = legal_nlp(text)
        ents = filter_legal_entities(doc)

        entity_list = []
        for ent in ents:
            entity_list.append(
                (ent.text.lower().strip(), ent.label_)
            )

        return entity_list

    except Exception as e:
        print(f"NER failed on generated text: {e}")
        return []


from sentence_transformers import SentenceTransformer
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

embed_model.eval()
for p in embed_model.parameters():
    p.requires_grad = False

embed_model = embed_model.to(device)
# embed_model = embed_model.half()

# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# embed_model = embed_model.to(torch.cuda.current_device())

def embed_entity_list(entity_list):
    if len(entity_list) == 0:
        return torch.empty((0, embed_model.get_sentence_embedding_dimension()), device=device)

    texts = [e[0] for e in entity_list]

    with torch.inference_mode():
        embeddings = embed_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    return embeddings

import math

from collections import defaultdict
import numpy as np


# DATASET_MEAN_DENSITY, DATASET_STD_DENSITY = compute_type_density_stats(train_entity_cache)

def canonicalize_generated_entities(entity_list):

    canonicalized = []

    for text, etype in entity_list:

        text = text.lower().strip()

        if etype == "PROVISION":
            canon = canonicalize_provision(text)
        elif etype == "PRECEDENT":
            canon = canonicalize_precedent(text)
        elif etype == "STATUTE":
            canon = canonicalize_statute(text)
        elif etype == "COURT":
            canon = canonicalize_court(text)
        else:
            canon = normalize_legal_text(text)

        # 🔥 CRITICAL FIX
        if canon is None:
            continue

        canonicalized.append((canon, etype))

    return canonicalized

# ============================================
# STRUCTURAL ENTITY REWARD
# ============================================


import re

from collections import defaultdict


def safe_reward_output(out):

    debug = out.get("debug", {})

    return {
        "reward": out.get("reward", 0.0),
        "coverage": debug.get("coverage", 0.0),
        "hallucination": debug.get("hallucination", 1.0),
        "type_metrics": debug.get("type_metrics", {})
    }

TYPE_IMPORTANCE = compute_type_importance(train_entity_cache)

def compute_structural_entity_reward(generated_text, cache, type_weights=None):

    # ================================
    # 1. LOAD ENTITIES
    # ================================
    source_entities = list(cache["source_metadata"].keys())
    source_emb = cache["source_embeddings"].to(device)


    # ================================
    # 2. EXTRACT + CANONICALIZE
    # ================================
    gen_entities = extract_generated_entities_spacy(generated_text)
    gen_entities = canonicalize_generated_entities(gen_entities)
    gen_entities = list(dict.fromkeys(gen_entities))

    # Penalize empty outputs
    if len(gen_entities) == 0:
        return {
            "reward": -0.25,
            "debug": {
                "coverage": 0.0,
                "hallucination": 1.0,
                "type_metrics": {}
            }
            }
    gen_emb = embed_entity_list(gen_entities).to(device)

    # ================================
    # 3. MATCHING
    # ================================
    matched_gen_src, matched_src = hybrid_match_v3(
        gen_entities, source_entities, gen_emb, source_emb
    )

    matched_gen_src = set(matched_gen_src)
    matched_src = set(matched_src)

    # ================================
    # 4. WEIGHTS
    # ================================
    def get_w(t):
        if type_weights is not None:
            return type_weights.get(t, TYPE_IMPORTANCE.get(t, 1.0))
        return TYPE_IMPORTANCE.get(t, 1.0)

    # ================================
    # 4. WEIGHTED SOURCE-ENTITY COVERAGE
    # ================================
    tp_cov = 0

    total_cov = 0

    for i, (_, t) in enumerate(source_entities):
        w = get_w(t)
        total_cov += w
        if i in matched_src:
            tp_cov += w

    coverage = tp_cov / (total_cov + 1e-8)

    
    
    # ================================
    # 8. EXPLICIT HALLUCINATION PENALTY
    # ================================
    hallucinated = sum(
        1 for i in range(len(gen_entities))
        if i not in matched_gen_src
    )

    hall_rate = hallucinated / (len(gen_entities) + 1e-8)

    alpha = 1.0
    gamma = 0.25

    reward = (
        alpha * coverage
        - gamma * hall_rate
    )


    # ================================
    # 9. TYPE METRICS
    # ================================
    from collections import defaultdict

    type_stats = defaultdict(lambda: {
        "source": 0,
        "generated": 0,
        "matched": 0
    })

    for _, t in source_entities:
        type_stats[t]["source"] += 1

    for _, t in gen_entities:
        type_stats[t]["generated"] += 1

    for idx in matched_src:
        t = source_entities[idx][1]
        type_stats[t]["matched"] += 1

    type_metrics = {}

    for t in TYPE_IMPORTANCE.keys():

        g = type_stats[t]["source"]
        p = type_stats[t]["generated"]
        m = type_stats[t]["matched"]

        coverage_t = m / (g + 1e-8)
        hallucination_t = 0.0

        if p > 0:
            hallucination_t = (p - m) / p

        type_metrics[t] = {
            "coverage": coverage_t,
            "hallucination": hallucination_t,
            "support": g,
            "gen_count": p
        }
    # ================================

    # 9. LENGTH REGULARIZATION (CRITICAL)

    # ================================

    target_len = 600
    length = len(generated_text.split())
    length_ratio = length / (target_len + 1e-8)
    length_penalty = abs(length_ratio - 1.0)
    # 🔥 apply penalty
    lambda_length = 0.08

    reward -= lambda_length * length_penalty

    return {
        "reward": reward,
        "debug": {
            "coverage": coverage,
            "hallucination": hall_rate,
            "type_metrics": type_metrics
        }
    }


import re

def extract_sample_id(prompt_text):
    match = re.search(r"\[SAMPLE_ID=(\d+)\]", prompt_text)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Sample ID not found in prompt")


def entity_reward(prompts, completions, completion_ids, **kwargs):

    rewards = []
    type_metrics = {}

    batch_size = len(completions)

    for i in range(batch_size):

        prompt_text = prompts[i]
        completion_text = completions[i]

        sample_id = extract_sample_id(prompt_text)

        if sample_id not in train_entity_cache:
            rewards.append(0.0)
            continue

        cache = train_entity_cache[sample_id]

        out = safe_reward_output(
            compute_structural_entity_reward(
                completion_text,
                cache,
                type_weights=TYPE_IMPORTANCE,
            )
        )

        reward = out["reward"]
        rewards.append(reward)

        tm = out["type_metrics"]

        for t, vals in tm.items():

            if t not in type_metrics:
                type_metrics[t] = {
                    "coverage": [],
                    "hallucination": [],
                    "support": [],
                    "gen_count": [],
                }

            type_metrics[t]["coverage"].append(vals["coverage"])
            type_metrics[t]["hallucination"].append(vals["hallucination"])
            type_metrics[t]["support"].append(vals["support"])
            type_metrics[t]["gen_count"].append(vals["gen_count"])

    rewards_tensor = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    )

    rewards_tensor = torch.tanh(rewards_tensor)
    rewards_tensor = torch.nan_to_num(
        rewards_tensor,
        nan=0.0,
    )

    def safe_mean(x):
        return sum(x) / len(x) if len(x) > 0 else 0.0

    log_dict = {
        "reward/mean": rewards_tensor.mean().item(),
        "reward/std": rewards_tensor.std().item(),
    }

    for t, vals in type_metrics.items():

        log_dict[f"type/{t}_coverage"] = safe_mean(
            vals["coverage"]
        )

        log_dict[f"type/{t}_hallucination"] = safe_mean(
            vals["hallucination"]
        )

        log_dict[f"type/{t}_support"] = safe_mean(
            vals["support"]
        )

        log_dict[f"type/{t}_gen_count"] = safe_mean(
            vals["gen_count"]
        )

        support = safe_mean(vals["support"])

        if support > 0:
            log_dict[f"type/{t}_gen_ratio"] = (
                safe_mean(vals["gen_count"]) / support
            )
        else:
            log_dict[f"type/{t}_gen_ratio"] = 0.0

    wandb.log(log_dict)

    return rewards_tensor

from transformers import TrainerCallback
import numpy as np
import wandb

class CollapseMonitorCallback(TrainerCallback):
    def __init__(self):
        self.window = 20  # moving window size

        self.kl_history = []
        self.reward_std_history = []
        self.length_history = []
        self.clipped_history = []
        self.intra_prompt_var_history = []  # 🔥 NEW

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is None:
            return

        kl = logs.get("kl")
        reward_std = logs.get("reward_std")# use your logged key
        completion_length = logs.get("completion_length")
        clipped_ratio = logs.get("completions/clipped_ratio")
        intra_prompt_var = logs.get("reward/intra_prompt_variance")  # 🔥 NEW

        if kl is not None:
            self.kl_history.append(kl)

        if reward_std is not None:
            self.reward_std_history.append(reward_std)

        if completion_length is not None:
            self.length_history.append(completion_length)

        if clipped_ratio is not None:
            self.clipped_history.append(clipped_ratio)

        if intra_prompt_var is not None:
            self.intra_prompt_var_history.append(intra_prompt_var)

        # Need enough history
        if len(self.kl_history) < self.window:
            return

        # -------------------------
        # Moving averages
        # -------------------------
        avg_kl = np.mean(self.kl_history[-self.window:])
        avg_reward_std = (
            np.mean(self.reward_std_history[-self.window:])
            if len(self.reward_std_history) >= self.window else 0.0
        )
        avg_length_std = (
            np.std(self.length_history[-self.window:])
            if len(self.length_history) >= self.window else 0.0
        )
        avg_clipped = (
            np.mean(self.clipped_history[-self.window:])
            if len(self.clipped_history) >= self.window else 0.0
        )
        avg_intra_prompt_var = (
            np.mean(self.intra_prompt_var_history[-self.window:])
            if len(self.intra_prompt_var_history) >= self.window else 0.0
        )

        # -------------------------
        # Collapse scoring
        # -------------------------
        collapse_score = 0

        # Excessive KL drift
        if avg_kl > 6.0:
            collapse_score += 1

        # Global reward variance collapsed
        if avg_reward_std < 0.01:
            collapse_score += 1

        # Generations all same length (mode collapse)
        if avg_length_std < 5:
            collapse_score += 1

        # Policy over-clipping
        if avg_clipped > 0.9:
            collapse_score += 1

        # 🔥 Intra-prompt collapse (very important)
        if avg_intra_prompt_var < 0.001:
            collapse_score += 1

        # -------------------------
        # Logging
        # -------------------------
        wandb.log({
            "monitor/avg_kl": avg_kl,
            "monitor/avg_reward_std": avg_reward_std,
            "monitor/avg_length_std": avg_length_std,
            "monitor/avg_clipped_ratio": avg_clipped,
            "monitor/avg_intra_prompt_variance": avg_intra_prompt_var,
            "monitor/collapse_score": collapse_score
        })

        # -------------------------
        # Hard stop condition
        # -------------------------
        if collapse_score >= 3:
            print("🚨 POLICY COLLAPSE DETECTED — STOPPING TRAINING")
            print(
                f"KL={avg_kl:.2f}, "
                f"RewardStd={avg_reward_std:.4f}, "
                f"LengthStd={avg_length_std:.2f}, "
                f"Clipped={avg_clipped:.2f}, "
                f"IntraVar={avg_intra_prompt_var:.6f}"
            )
            control.should_training_stop = True
            return control
# %%

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    temperature = 0.9,
    top_p = 0.98,
    min_p = 0.05,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)



from trl import GRPOConfig, GRPOTrainer
max_prompt_length=max_seq_length
max_completion_length=900
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    learning_rate = 2e-6, ### earlier it was 4e-6, but reduced to stabilize training with the new reward and resume from checkpoint
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    num_generations = 4,
    gradient_accumulation_steps = 4,
    max_prompt_length = 4500,
    max_completion_length = max_completion_length,
    num_train_epochs = 5, # Set to 1 for a full training run
    # max_steps = 500,
    beta=0.01,  # ADD THIS
    save_steps = 250,
    max_grad_norm = 1.0,
    output_dir = "llama_0shot_rl_modified_entity_reward_v2",
    report_to="wandb",
    temperature = 0.9,
    top_p = 0.98,
    # min_completion_length = 600
)

# %%
import torch
from tqdm import tqdm

@torch.inference_mode()
def evaluate_on_validation(
    model,
    tokenizer,
    val_dataset,
    val_entity_cache,
    max_new_tokens=900,
):
    model.eval()

    total_reward = 0.0
    total_samples = 0

    coverage_total = 0.0
    hallucination_total = 0.0
    length_total = 0.0

    for sample in val_dataset:

        sample_id = sample["id"]
        prompt = sample["prompt"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=200,
            do_sample=True,
            temperature=0.9,
            top_p=0.98,
        )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        generated_text = tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True,
        )

        del outputs, gen_tokens, inputs
        torch.cuda.empty_cache()

        if sample_id not in val_entity_cache:
            continue

        out = safe_reward_output(
            compute_structural_entity_reward(
                generated_text,
                val_entity_cache[sample_id],
                type_weights=TYPE_IMPORTANCE,
            )
        )

        reward = math.tanh(out["reward"])

        if math.isnan(reward):
            reward = 0.0

        coverage = out["coverage"]
        hallucination = out["hallucination"]

        length = len(generated_text.split())

        total_reward += float(reward)
        coverage_total += coverage
        hallucination_total += hallucination
        length_total += length

        total_samples += 1

    if total_samples == 0:
        return {
            "mean_reward":0.0,
            "coverage":0.0,
            "hallucination":1.0,
            "score":0.0,
        }

    avg_coverage = coverage_total / total_samples
    avg_hallucination = hallucination_total / total_samples
    avg_len = length_total / total_samples
    mean_reward = total_reward / total_samples

    print(f"\nValidation Reward: {mean_reward:.4f}\n")

    wandb.log({
        "val/mean_reward": mean_reward,
        "val/coverage": avg_coverage,
        "val/hallucination": avg_hallucination,
        "val/length_mean": avg_len,
    })

    return {
        "mean_reward": mean_reward,
        "coverage": avg_coverage,
        "hallucination": avg_hallucination,
        "score": mean_reward,
    }
    

from collections import defaultdict
from tqdm import tqdm

from transformers import TrainerCallback

class ValidationRewardEarlyStopCallback(TrainerCallback):

    def __init__(
        self,
        trainer,
        val_dataset,
        val_entity_cache,
        eval_steps=250,
        patience=5,
        min_delta=0.005,
    ):

        self.trainer_ref = trainer
        self.val_dataset = val_dataset
        self.val_entity_cache = val_entity_cache
        self.eval_steps = eval_steps
        self.patience = patience
        self.min_delta = min_delta

        self.best_reward = -float("inf")
        self.bad_evals = 0
        self.trainer_ref = None  # will be set later

    # def on_train_begin(self, args, state, control, **kwargs):
    #     self.trainer_ref = kwargs.get("trainer", None)

    # def on_init_end(self, args, state, control, **kwargs):
    #     self.trainer_ref = kwargs.get("trainer", None)

    def on_step_end(self, args, state, control, **kwargs):

        if state.global_step % self.eval_steps != 0:
            return

        if self.trainer_ref is None:
            raise RuntimeError("Trainer reference not set — callback misconfigured")

        model = self.trainer_ref.model
        tokenizer = self.trainer_ref.processing_class

        print(f"\n🔎 Running validation at step {state.global_step}")

        results = evaluate_on_validation(
            model,
            tokenizer,
            self.val_dataset,
            self.val_entity_cache,
            max_new_tokens=900,
        )

        val_reward = results["mean_reward"]

        if val_reward > self.best_reward + self.min_delta:

            self.best_reward = val_reward
            self.bad_evals = 0

            print(
                f"New best validation reward: "
                f"{val_reward:.4f}"
            )

        else:

            self.bad_evals += 1

            print(
                f"⚠ No improvement. "
                f"Best={self.best_reward:.4f}, "
                f"Current={val_reward:.4f}, "
                f"BadEvals={self.bad_evals}/{self.patience}"
            )

        wandb.log({
            "early_stop/validation_reward": val_reward,
            "early_stop/best_reward": self.best_reward,
        })
        if self.bad_evals >= self.patience:
            print("🛑 Early stopping: validation reward plateau.")
            control.should_training_stop = True
            return control


from transformers import TrainerCallback

class FixedSampleTrackingCallback(TrainerCallback):

    def __init__(
        self,
        tokenizer,
        val_entity_cache,
        sample_ids,
        val_lookup,
        eval_steps=250,
    ):
        self.tokenizer = tokenizer
        self.val_entity_cache = val_entity_cache
        self.sample_ids = sample_ids
        self.eval_steps = eval_steps
        self.val_lookup = val_lookup
        self.trainer_ref = None

    # def on_init_end(self, args, state, control, **kwargs):
    #     self.trainer_ref = kwargs.get("trainer", None)

    def _generate_and_log(self, model, step):

        print(f"\n📊 SAMPLE GENERATION @ step {step}\n")

        for sid in self.sample_ids:

            sample = self.val_lookup.get(sid)
            if sample is None:
                continue

            prompt = sample["prompt"]

            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.7,
                top_p=0.98
            )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]

            text = self.tokenizer.decode(
                gen_tokens,
                skip_special_tokens=True
            )

            out = safe_reward_output(
                    compute_structural_entity_reward(
                    text,
                    self.val_entity_cache[sid],
                    type_weights=TYPE_IMPORTANCE,
                )
            )

            reward = out["reward"]
            coverage = out["coverage"]
            hall = out["hallucination"]

            print(
                f"[ID {sid}] "
                f"reward={reward:.3f} | "
                f"coverage={coverage:.3f} | "
                f"hallucination={hall:.3f}"
            )

            print(text[:800])
            print("=" * 80)

            if "wandb" in globals():
                wandb.log({
                    f"sample_{sid}/reward": reward,
                    f"sample_{sid}/coverage": coverage,
                    f"sample_{sid}/hallucination": hall,
                    f"sample_{sid}/text": text[:1000],
                    "step": step,
                })

            del outputs, gen_tokens, inputs
            torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):

        if state.global_step % self.eval_steps != 0:
            return

        if self.trainer_ref is None:
            raise RuntimeError("Trainer reference not set — callback misconfigured")

        model = self.trainer_ref.model
        model.eval()

        self._generate_and_log(model, state.global_step)

    def on_save(self, args, state, control, **kwargs):
        """🔥 Generate samples at checkpoint save"""

        if self.trainer_ref is None:
            raise RuntimeError("Trainer reference not set — callback misconfigured")

        model = self.trainer_ref.model
        model.eval()

        print(f"\n💾 CHECKPOINT SAVE @ step {state.global_step}")
        self._generate_and_log(model, state.global_step)

fixed_sample_ids = [2, 3, 6, 10, 12]

val_subset = val_dataset.select(range(50))
val_lookup = {x["id"]: x for x in val_subset}

class DiagnosticRunnerCallback(TrainerCallback):

    def __init__(
        self,
        tokenizer,
        dataset,
        entity_cache,
        extract_fn,
        canonicalize_fn,
        embed_fn,
        match_fn,
        eval_steps=250,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.entity_cache = entity_cache
        self.extract_fn = extract_fn
        self.canonicalize_fn = canonicalize_fn
        self.embed_fn = embed_fn
        self.match_fn = match_fn
        self.eval_steps = eval_steps

        self.trainer_ref = None

    # def on_init_end(self, args, state, control, **kwargs):
    #     self.trainer_ref = kwargs.get("trainer", None)

    def on_step_end(self, args, state, control, **kwargs):

        if state.global_step % self.eval_steps != 0:
            return

        if self.trainer_ref is None:
            raise RuntimeError("Trainer reference not set — callback misconfigured")

        model = self.trainer_ref.model
        model.eval()

        print(f"\n\n🧪 DIAGNOSTIC @ step {state.global_step}\n")

        diagnostic_10_samples(
            llm_model=model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            entity_cache=self.entity_cache,
            extract_fn=self.extract_fn,
            canonicalize_fn=self.canonicalize_fn,
            embed_fn=self.embed_fn,
            match_fn=self.match_fn,
        )

        torch.cuda.empty_cache()


def reward_with_step(*args, **kwargs):
    return entity_reward(
        *args,
        global_step=trainer.state.global_step,
        **kwargs
    )

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_with_step],
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[
    ValidationRewardEarlyStopCallback(
        trainer=None,  # temporary placeholder
        val_dataset=val_subset,
        val_entity_cache=val_entity_cache,
        eval_steps=250,
        patience=5,
        min_delta=0.005,
        ),
        CollapseMonitorCallback(),
        # 🔥 FIXED sample tracking (will now work)

        FixedSampleTrackingCallback(
            tokenizer=tokenizer,
            val_entity_cache=val_entity_cache,
            sample_ids=fixed_sample_ids,
            eval_steps=250,
            val_lookup=val_lookup,
        ),

        # 🚀 NEW DIAGNOSTIC CALLBACK

        DiagnosticRunnerCallback(
            tokenizer=tokenizer,
            dataset=train_dataset,
            entity_cache=train_entity_cache,
            extract_fn=extract_generated_entities_spacy,
            canonicalize_fn=canonicalize_generated_entities,
            embed_fn=embed_entity_list,
            match_fn=hybrid_match_v3,
            eval_steps=250,
        ),
    ],
)

for cb in trainer.callback_handler.callbacks:
    if hasattr(cb, "trainer_ref"):
        cb.trainer_ref = trainer




@torch.no_grad()
def diagnostic_10_samples(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    match_fn,
    n=10
):
    print("\n========== 10-SAMPLE DIAGNOSTIC ==========\n")

    summary = {
    "total_source": 0,
    "total_matched": 0,
    "reward": [],
    "coverage": [],
    "hallucination": [],
    "generation_failure": 0,
    "matching_failure": 0,
    "good_match": 0,
}

    for i in range(min(n, len(dataset))):

        sample = dataset[i]
        sid = sample["id"]

        inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7
        )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # --------------------------
        # Entities
        # --------------------------
        gen_entities = canonicalize_fn(extract_fn(text))
        gen_entities = list(dict.fromkeys(gen_entities))

        source_entities = list(entity_cache[sid]["source_metadata"].keys())

        if len(source_entities) == 0:
            continue

        gen_emb = embed_fn(gen_entities)
        source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)

        matched_gen, matched_source = match_fn(
            gen_entities,
            source_entities,
            gen_emb,
            source_emb
        )

        matched_gen = set(matched_gen)
        matched_source = set(matched_source)

        matched_pairs = list(zip(matched_gen, matched_source))

        out = safe_reward_output(
            compute_structural_entity_reward(
                text,
                entity_cache[sid],
                type_weights=TYPE_IMPORTANCE,
            )
        )

        reward = out["reward"]
        coverage = out["coverage"]
        hallucination = out["hallucination"]

        # --------------------------
        # Analysis per sample
        # --------------------------
        print(f"\n--- SAMPLE {sid} ---")
        print(f"Reward        : {reward:.3f}")
        print(f"Coverage      : {coverage:.3f}")
        print(f"Hallucination : {hallucination:.3f}")
        print(f"Source ents   : {len(source_entities)}")
        print(f"Generated ents: {len(gen_entities)}")
        print(f"Matched ents  : {len(matched_source)}")

        # Type-wise stats
        type_stats = {}

        for _, t in source_entities:
            type_stats.setdefault(t, {"source": 0, "generated": 0, "matched": 0})
            type_stats[t]["source"] += 1

        for _, t in gen_entities:
            type_stats.setdefault(t, {"source": 0, "generated": 0, "matched": 0})
            type_stats[t]["generated"] += 1

        for idx in matched_source:
            t = source_entities[idx][1]
            type_stats[t]["matched"] += 1

        failure_flag = False

        for t in ["STATUTE", "PROVISION", "PRECEDENT"]:
            s = type_stats.get(t, {"source":0,"generated":0,"matched":0})

            print(
                f"{t:10s} | src={s['source']} "
                f"gen={s['generated']} "
                f"match={s['matched']}"
            )

            if s["source"] > 0:

                # GENERATION FAILURE
                if s["generated"] == 0:
                    print(f"Missing entity type in generation → {t}")
                    summary["generation_failure"] += 1
                    failure_flag = True

                # MATCHING FAILURE
                elif s["matched"] == 0:
                    print(f"Generated entity could not be aligned → {t}")
                    summary["matching_failure"] += 1
                    failure_flag = True

                else:
                    summary["good_match"] += 1

        # --------------------------
        # 🔍 Detailed debugging
        # --------------------------
        if failure_flag:

            print("\n🔎 DEBUG DETAILS:")

            # Missing types
            missing_types = [
                t for t in ["STATUTE", "PROVISION", "PRECEDENT"]
                if type_stats.get(t, {}).get("source", 0) > 0
                and type_stats.get(t, {}).get("generated", 0) == 0
            ]

            if missing_types:
                print("Missing types:", missing_types)

            # Generated entities
            print("\nGenerated entities:")
            for e in gen_entities:
                print("  ", e)

            # Source entities (subset for readability)
            print("\nSource entities (first 10):")
            for e in source_entities[:10]:
                print("  ", e)

            # Matched pairs
            print("\nMatched pairs:")
            for gi, si in matched_pairs:
                print("  GEN:", gen_entities[gi], " <--> SRC:", source_entities[si])

            # Show generation text snippet
            print("\nGenerated text (snippet):")
            print(text[:400])

        summary["reward"].append(reward)
        summary["coverage"].append(coverage)
        summary["hallucination"].append(hallucination)
        summary["total_source"] += len(source_entities)
        summary["total_matched"] += len(matched_source)

    # --------------------------
    # FINAL DIAGNOSIS
    # --------------------------
    print("\n========== FINAL DIAGNOSIS ==========\n")

    print("Total Source Entities:", summary["total_source"])
    print("Total Matched:", summary["total_matched"])

    coverage = summary["total_matched"] / max(summary["total_source"], 1)

    print(f"Average Reward       : {np.mean(summary['reward']):.3f}")
    print(f"Average Coverage     : {np.mean(summary['coverage']):.3f}")
    print(f"Average Hallucination: {np.mean(summary['hallucination']):.3f}")

    overall_entity_coverage = (
        summary["total_matched"] /
        max(summary["total_source"], 1)
    )

    print(f"Entity Coverage      : {overall_entity_coverage:.3f}")

    print("\nFailure Breakdown:")
    print("Generation Failure:", summary["generation_failure"])
    print("Matching Failure  :", summary["matching_failure"])
    print("Good Matches      :", summary["good_match"])

    print("\n=====================================\n")


# for i in range(10):
#     sample = train_dataset[i]

#     sid = sample["id"]
#     cache = train_entity_cache[sid]

#     print("SOURCE TEXT START:")
#     print(cache["source_text"][:300])
#     print("\nPROMPT START:")
#     print(sample["prompt"][:600])
#     print("="*80)



# run_all_debug(
#     llm_model=model,
#     tokenizer=tokenizer,
#     dataset=train_dataset,
#     entity_cache=train_entity_cache,
#     compute_reward_fn=compute_structural_entity_reward,
#     extract_fn=extract_generated_entities_spacy,
#     canonicalize_fn=canonicalize_generated_entities,
#     embed_fn=embed_entity_list,
#     debug_similarity_matrix=debug_similarity_matrix_v3,
#     match_fn=hybrid_match_v3,
# )

print("===== PRE-TRAIN DIAGNOSTIC =====")
diagnostic_10_samples(
    llm_model=model,
    tokenizer=tokenizer,
    dataset=train_dataset,
    entity_cache=train_entity_cache,
    extract_fn=extract_generated_entities_spacy,
    canonicalize_fn=canonicalize_generated_entities,
    embed_fn=embed_entity_list,
    match_fn=hybrid_match_v3,
)



trainer.train(resume_from_checkpoint="llama_0shot_rl_modified_entity_reward_v2/checkpoint-2750")



