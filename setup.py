import os

HF_TOKEN   = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # ← YOUR TOKEN
HUB_REPO   = "dschauhan08/qwen-reasoning-finetuned"      # ← YOUR REPO
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

SEQ_LEN          = 4096
BATCH_PER_DEVICE = 1
GRAD_ACCUM       = 4      # Effective = 1 × 2 GPUs × 4 = 8 sequences/step

LORA_R       = 16
LORA_ALPHA   = 16
LORA_DROPOUT = 0.0

LR           = 1e-4
WARMUP_STEPS = 1500
TRAIN_STEPS  = 30_000
WEIGHT_DECAY = 0.01

SAVE_EVERY_STEPS  = 300
KEEP_CHECKPOINTS  = 3
KEEP_HUB_CKPTS    = 3
MAX_ROWS_PER_SOURCE = 15_000
SEED       = 42
OUTPUT_DIR = "./qwen_train_out"
# ── Dataset sources ────────────────────────────────────────────────────────
DATASET_SOURCES = [

    # ── Claude Opus 4.6 reasoning ─────────────────────────────────────────
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "Crownelius/Opus-4.6-Reasoning-2100x-formatted",
    "dalisoft/claude-opus-4.6-high-reasoning-700x",
    "Roman1111111/claude-opus-4.6-10000x",

    # ── NVIDIA ────────────────────────────────────────────────────────────
    {"name": "nvidia/OpenCodeReasoning-2", "split": "python"},
    {"name": "nvidia/OpenCodeReasoning-2", "split": "cpp"},
    # ── Other reasoning ───────────────────────────────────────────────────
    "Roman1111111/gemini-3.1-pro-hard-high-reasoning",
    "Roman1111111/gpt-5.4-step-by-step-reasoning",
    # ── Mixed style (lower caps to avoid dialect dilution) ────────────────
    "Crownelius/GLM-5.0-8000x-formatted-fixed",
    "Crownelius/Opus-4.5-3000x-formatted",
    "Crownelius/Gemini-3-Pro-Opus-4.5-Kimi-K2.5-13000x-formatted",

    # ── tandevllc offsec ──────────────────────────────────────────────────
    {"name": "tandevllc/offsec_redteam_codes", "config": "penetration-testing"},
    {"name": "tandevllc/offsec_redteam_codes", "config": "malware-analysis"},
    {"name": "tandevllc/offsec_redteam_codes", "config": "ethical-hacking"},
]

STYLE_CAPS = {
    "Crownelius/GLM-5.0-8000x-formatted-fixed":                   3_000,
    "Crownelius/Gemini-3-Pro-Opus-4.5-Kimi-K2.5-13000x-formatted": 3_000,
    "tandevllc/offsec_redteam_codes":                             2_500,
}

os.environ["WANDB_DISABLED"]          = "true"
os.environ["HF_TOKEN"]                = HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("✅  Config loaded.")
print(f"    Model   : {MODEL_NAME}")
print(f"    SEQ_LEN : {SEQ_LEN}")
print(f"    Steps   : {TRAIN_STEPS:,}  (warmup: {WARMUP_STEPS})")
print(f"    Sources : {len(DATASET_SOURCES)}")
