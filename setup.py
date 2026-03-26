# ── HuggingFace credentials ──────────────────────────────────────────
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # ← YOUR TOKEN HERE
HUB_REPO = "dschauhan08/qwen-reasoning-finetuned"      # ← YOUR REPO HERE

# ── Base model ───────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-4B"

# ── Sequence length ──────────────────────────────────────────────────
SEQ_LEN = 4096   # Maxed out. VRAM will hover right at the 14GB-15GB edge.

# ── Batch & accumulation ─────────────────────────────────────────────
BATCH      = 1   # STRICTLY 1 to prevent OOM at 8192 context.
GRAD_ACCUM = 8   

# ── LoRA settings ────────────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 16   
LORA_DROPOUT = 0.0  

# ── Learning rate ────────────────────────────────────────────────────
LR            = 2e-5   
WARMUP_STEPS  = 100    
TRAIN_STEPS   = 15000  

# ── Checkpoint cadence ───────────────────────────────────────────────
SAVE_EVERY_STEPS = 300    # Save locally frequently
UPLOAD_INTERVAL  = 21600  # Upload to HF every 6 hours
KEEP_SNAPSHOTS   = 3      

# ── Misc ─────────────────────────────────────────────────────────────
SEED = 42
