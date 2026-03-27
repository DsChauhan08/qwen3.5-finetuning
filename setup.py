import os
 
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # ← YOUR TOKEN HERE
HUB_REPO = "dschauhan08/qwen-reasoning-finetuned"      # ← YOUR REPO HERE
 
MODEL_NAME = "Qwen/Qwen3.5-4B"
 
SEQ_LEN          = 4500   # Safe for 2×T4 DDP. Fits ~85% of reasoning traces untruncated.
BATCH_PER_DEVICE = 1      # 1 sequence per GPU per micro-step.
GRAD_ACCUM       = 4      # Effective batch = 1 GPU × 2 × 4 accum = 8 seq/step
 
LORA_R       = 16
LORA_ALPHA   = 16     # = R → scale factor 1.0
LORA_DROPOUT = 0.0    # Must be 0 — dropout destabilises GatedDeltaNet paths
 
LR           = 1e-4
WARMUP_STEPS = 1500   # 5 % of 30k — safe warm-up ramp
TRAIN_STEPS  = 30_000 # ~1.95 epochs over the ~123k-row dataset
WEIGHT_DECAY = 0.01
 
SAVE_EVERY_STEPS  = 300   # ~77 min at 15.5 sec/step
KEEP_CHECKPOINTS  = 3     # Local disk limit
KEEP_HUB_CKPTS    = 3     # Hub copies pruned to this after each push
 
MAX_ROWS_PER_SOURCE = 15_000
SEED       = 42
OUTPUT_DIR = "./qwen_train_out"
 
# ── Dataset sources ────────────────────────────────────────────────────────
# nvidia/OpenCodeReasoning-2 appears ONCE (was duplicated in previous version
# which wasted 15k rows and biased the model toward that single source)
DATASET_SOURCES = [
    # High-priority reasoning — pure, well-formatted, same model family
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "Crownelius/Opus-4.6-Reasoning-2100x-formatted",
    "dalisoft/claude-opus-4.6-high-reasoning-700x",
    # NVIDIA — strong quality control, diverse reasoning + code
    "nvidia/OpenCodeReasoning-2",
    "nvidia/Nemotron-SFT-Competitive-Programming-v2",
    "nvidia/Nemotron-Terminal-Synthetic-Tasks",
    "nvidia/Nemotron-RL-ReasoningGym-v1",
    # Broader reasoning corpus
    "Roman1111111/claude-opus-4.6-10000x",
    "Roman1111111/gemini-3.1-pro-hard-high-reasoning",
    "Roman1111111/gpt-5.4-step-by-step-reasoning",
    "artillerywu/DeepResearch-9K",
    # Mixed / agentic (style-capped to avoid dialect dilution)
    "Crownelius/Opus-4.5-3000x-formatted",
    "Crownelius/Gemini-3-Pro-Opus-4.5-Kimi-K2.5-13000x-formatted",
    "Crownelius/GLM-5.0-8000x-formatted-fixed",
    "Crownelius/Agentic-SFT-1000x",
    # Code + security (user request)
    "tandevllc/offsec_redteam_codes",
]
 
# Sources whose style could dilute Qwen3.5's native reasoning dialect
# get a lower row cap than MAX_ROWS_PER_SOURCE.
STYLE_CAPS = {
    "Crownelius/Gemini-3-Pro-Opus-4.5-Kimi-K2.5-13000x-formatted": 3_000,
    "Crownelius/GLM-5.0-8000x-formatted-fixed": 3_000,
}
 
os.environ["WANDB_DISABLED"]         = "true"
os.environ["HF_TOKEN"]               = HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
print("✅  Config loaded.")
print(f"    Model      : {MODEL_NAME}")
print(f"    SEQ_LEN    : {SEQ_LEN}")
print(f"    Steps      : {TRAIN_STEPS:,}  (warmup: {WARMUP_STEPS})")
print(f"    Sources    : {len(DATASET_SOURCES)}")
 
