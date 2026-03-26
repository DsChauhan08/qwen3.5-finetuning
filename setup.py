# ── HuggingFace credentials ──────────────────────────────────────────
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # ← YOUR TOKEN HERE
HUB_REPO = "dschauhan08/qwen-reasoning-finetuned"      # ← YOUR REPO HERE

# ── Base model ───────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-4B"

# ── Sequence length ──────────────────────────────────────────────────
SEQ_LEN = 4096   # Maxed out. VRAM will hover right at the 14GB-15GB edge.

# ── Batch & accumulation ─────────────────────────────────────────────
BATCH      = 1   # STRICTLY 1 to prevent OOM at 4096 context.
GRAD_ACCUM = 8   # Global Batch Size = 8

# ── LoRA settings ────────────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 16   # 1x scaling to safely integrate the aggressive red-team/agentic data
LORA_DROPOUT = 0.0  

# ── Learning rate ────────────────────────────────────────────────────
LR            = 2e-4   # Optimized learning rate for LoRA adapters
WARMUP_STEPS  = 1000   # ~3.5% of total steps for a stable, safe ramp-up
TRAIN_STEPS   = 27500  # Yields ~1.62 epochs over the 135k capped dataset

# ── Checkpoint cadence ───────────────────────────────────────────────
SAVE_EVERY_STEPS = 300    # Save locally roughly every 2,400 rows
UPLOAD_INTERVAL  = 21600  # Upload to HF every 6 hours
KEEP_SNAPSHOTS   = 3      

# ── Misc ─────────────────────────────────────────────────────────────
SEED = 42
