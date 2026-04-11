# CLEAN INSTALL (reliable)
!pip -q install -U uv

!pip -q uninstall -y \
  unsloth transformers tokenizers accelerate peft trl bitsandbytes \
  datasets huggingface_hub torchvision torchaudio torch triton

!pip -q cache purge || true

# PyTorch (matched set)
!uv pip install --system --upgrade --force-reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

# Core libs (install datasets explicitly here)
!uv pip install --system --upgrade --force-reinstall \
  datasets huggingface_hub trl peft accelerate bitsandbytes

# Transformers + Unsloth (latest)
!uv pip install --system --upgrade --force-reinstall \
  "transformers @ git+https://github.com/huggingface/transformers.git" \
  "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"


# =========================================================
# CELL 2 — TRAINING
# =========================================================
import os
import re
import json
import time
import shutil
from pathlib import Path

import torch
from kaggle_secrets import UserSecretsClient
from datasets import load_dataset
from huggingface_hub import login, HfApi, snapshot_download
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

print("Torch:", torch.__version__)
print("Torchvision:", __import__("torchvision").__version__)
print("Transformers:", __import__("transformers").__version__)
print("CUDA available:", torch.cuda.is_available())

# ----------------------------
# AUTH
# ----------------------------
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN secret not found in Kaggle secrets.")
login(token=hf_token)

# ----------------------------
# ENV
# ----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_MODEL_ID = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DATASET_ID = "dschauhan08/mega-reasoning-unsloth"
HUB_REPO_ID = "dschauhan08/gemma-4-e2b-reasoning"

WORKDIR = Path("/kaggle/working")
LOCAL_CKPT_DIR = WORKDIR / "checkpoints"
FINAL_DIR = WORKDIR / "final_model"
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Stable Kaggle defaults
MAX_SEQ_LENGTH = 2048   # increase to 4096 only if memory stays clean
NUM_EPOCHS = 2
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 4
LR = 1e-4
WARMUP_RATIO = 0.03
EVAL_STEPS = 250
SAVE_STEPS = 250
LOG_STEPS = 10
KEEP_LAST_LOCAL_CKPTS = 2
SEED = 3407

# ----------------------------
# HUB REPO
# ----------------------------
api = HfApi()
api.create_repo(
    repo_id=HUB_REPO_ID,
    repo_type="model",
    private=True,
    exist_ok=True,
)

# ----------------------------
# RESUME FROM LATEST HUB CHECKPOINT
# ----------------------------
def latest_hub_checkpoint(repo_id: str):
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"Could not list repo files: {e}")
        return None

    steps = []
    for f in files:
        m = re.match(r"checkpoint-(\d+)/trainer_state\.json$", f)
        if m:
            steps.append(int(m.group(1)))

    if not steps:
        return None

    latest_step = max(steps)
    ckpt_name = f"checkpoint-{latest_step}"
    local_ckpt_path = LOCAL_CKPT_DIR / ckpt_name

    if not local_ckpt_path.exists():
        print(f"Downloading latest checkpoint from Hub: {ckpt_name}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(LOCAL_CKPT_DIR),
            local_dir_use_symlinks=False,
            allow_patterns=[f"{ckpt_name}/**"],
            resume_download=True,
            max_workers=8,
        )

    return str(local_ckpt_path)

print("Checking Hub for existing checkpoints...")
resume_checkpoint_path = latest_hub_checkpoint(HUB_REPO_ID)

if resume_checkpoint_path:
    print(f"Resuming from: {resume_checkpoint_path}")
else:
    print("No checkpoint found. Starting fresh.")

# ----------------------------
# LOAD BASE MODEL
# ----------------------------
print(f"Loading base model: {BASE_MODEL_ID}")
start = time.time()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
)

print(f"Base model loaded in {time.time() - start:.1f}s")

# ----------------------------
# LORA
# ----------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# ----------------------------
# DATASET
# ----------------------------
ds = load_dataset(DATASET_ID)

if "validation" in ds:
    dataset_train = ds["train"]
    dataset_val = ds["validation"]
else:
    split = ds["train"].train_test_split(test_size=0.02, seed=SEED)
    dataset_train = split["train"]
    dataset_val = split["test"]

print("Train columns:", dataset_train.column_names)
print("Val columns:", dataset_val.column_names)

if "messages" not in dataset_train.column_names:
    raise ValueError("Expected 'messages' column in train split.")
if "messages" not in dataset_val.column_names:
    raise ValueError("Expected 'messages' column in validation split.")

def _maybe_json(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def format_prompts(batch):
    texts = []
    eos = tokenizer.eos_token or ""

    for chat in batch["messages"]:
        normalized = []

        for msg in chat:
            role = (msg.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"

            content = msg.get("content", "")
            if content is None:
                content = ""

            if role == "assistant":
                reasoning = msg.get("reasoning")
                tools = msg.get("tools") or msg.get("tool_calls")

                if reasoning:
                    content = f"<think>\n{reasoning}\n</think>\n{content}"

                if tools:
                    tools = _maybe_json(tools)
                    content = f"<tool_call>\n{tools}\n</tool_call>\n{content}"

            normalized.append({"role": role, "content": content})

        text = tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=False,
        )

        if eos and not text.endswith(eos):
            text += eos

        texts.append(text)

    return {"text": texts}

print("Formatting dataset...")
dataset_train = dataset_train.map(
    format_prompts,
    batched=True,
    num_proc=2,
    remove_columns=dataset_train.column_names,
    desc="Formatting train",
)
dataset_val = dataset_val.map(
    format_prompts,
    batched=True,
    num_proc=2,
    remove_columns=dataset_val.column_names,
    desc="Formatting val",
)
print("Formatting complete.")

# ----------------------------
# CHECKPOINT CALLBACK
# ----------------------------
class SavePushDeleteCallback(TrainerCallback):
    def __init__(self, repo_id, keep_last_n_local=2):
        self.repo_id = repo_id
        self.api = HfApi()
        self.keep_last_n_local = keep_last_n_local

    def on_save(self, args, state, control, **kwargs):
        ckpt_name = f"checkpoint-{state.global_step}"
        ckpt_path = os.path.join(args.output_dir, ckpt_name)
        if not os.path.isdir(ckpt_path):
            return control

        print(f"\nUploading {ckpt_name} to Hub...")
        self.api.upload_folder(
            folder_path=ckpt_path,
            repo_id=self.repo_id,
            repo_type="model",
            path_in_repo=ckpt_name,
        )
        print("Upload complete.")

        found = []
        for d in os.listdir(args.output_dir):
            m = re.match(r"checkpoint-(\d+)$", d)
            full = os.path.join(args.output_dir, d)
            if m and os.path.isdir(full):
                found.append((int(m.group(1)), full))

        found.sort(key=lambda x: x[0])
        old = found[:-self.keep_last_n_local]

        for _, old_path in old:
            try:
                shutil.rmtree(old_path)
                print(f"Removed local checkpoint: {old_path}")
            except Exception as e:
                print(f"Could not remove {old_path}: {e}")

        return control

# ----------------------------
# TRAINER
# ----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        output_dir=str(LOCAL_CKPT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=LOG_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=SEED,
        neftune_noise_alpha=5,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    ),
    callbacks=[SavePushDeleteCallback(HUB_REPO_ID, KEEP_LAST_LOCAL_CKPTS)],
)

# ----------------------------
# TRAIN
# ----------------------------
print("Starting training...")
if resume_checkpoint_path:
    trainer.train(resume_from_checkpoint=resume_checkpoint_path)
else:
    trainer.train()

# ----------------------------
# FINAL SAVE + PUSH
# ----------------------------
print("Saving final artifacts...")
trainer.model.save_pretrained(str(FINAL_DIR))
tokenizer.save_pretrained(str(FINAL_DIR))

print("Uploading final artifacts to Hub...")
api.upload_folder(
    folder_path=str(FINAL_DIR),
    repo_id=HUB_REPO_ID,
    repo_type="model",
    path_in_repo="",
)

print("Done.")
