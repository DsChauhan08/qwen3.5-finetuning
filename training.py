
import gc
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, login, snapshot_download
from transformers import AutoTokenizer

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"\nPyTorch : {torch.__version__}  |  CUDA : {torch.version.cuda}")
print(f"GPUs    : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB")

# ── HF auth ───────────────────────────────────────────────────────────────
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(HUB_REPO, private=True, exist_ok=True)
    print(f"\n✅  HF repo: https://huggingface.co/{HUB_REPO}")
except Exception as e:
    print(f"ℹ️   Repo note: {e}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
#  RESUME: download the latest checkpoint from HF Hub
#
#  hub_strategy="checkpoint" (set in Cell 4's SFTConfig) pushes two things
#  after every SAVE_EVERY_STEPS steps:
#    1. checkpoint-{step}/   (individual numbered folder — like every_save)
#    2. last-checkpoint/     (always the most recent, for easy resuming)
#
#  We try last-checkpoint/ first (guaranteed to be the newest) and fall
#  back to scanning for checkpoint-N/ patterns (safety net).
# ─────────────────────────────────────────────────────────────────────────

def hub_resume() -> bool:
    """
    Finds and downloads the latest training checkpoint from HF Hub.
    Returns True if a usable checkpoint was downloaded.

    Strategy:
      1. Try last-checkpoint/  (maintained by hub_strategy="checkpoint")
      2. Fall back to highest-numbered checkpoint-N/
    """
    print("\n🔍  Checking HF Hub for latest checkpoint…")
    try:
        all_files = list(api.list_repo_files(HUB_REPO))
    except Exception as e:
        print(f"    Hub unreachable ({e}) — starting fresh")
        return False

    if not all_files:
        print("    Repo is empty — starting fresh")
        return False

    def _download(folder_pattern: str, display_name: str) -> bool:
        """Download a specific folder from the hub into OUTPUT_DIR."""
        matching = [f for f in all_files if f.startswith(f"{folder_pattern}/")]
        if not matching:
            return False
        print(f"    Downloading {display_name}…")
        try:
            snapshot_download(
                repo_id        = HUB_REPO,
                allow_patterns = [f"{folder_pattern}/*"],
                local_dir      = OUTPUT_DIR,
                token          = HF_TOKEN,
            )
            state_path = Path(OUTPUT_DIR) / folder_pattern / "trainer_state.json"
            if not state_path.exists():
                print(f"    ⚠️  trainer_state.json not found in {folder_pattern} — corrupt?")
                return False
            step = json.loads(state_path.read_text()).get("global_step", "?")
            print(f"    ✅  Checkpoint ready at optimiser step {step}")
            return True
        except Exception as e:
            print(f"    Download failed: {e}")
            return False

    # ── Option 1: last-checkpoint/ (hub_strategy="checkpoint" always keeps this) ──
    if _download("last-checkpoint", "last-checkpoint/"):
        # Trainer.train(resume_from_checkpoint=True) looks for the
        # LATEST local checkpoint-N/ folder. We need to rename
        # last-checkpoint/ to a checkpoint-N/ name so Trainer finds it.
        state_path = Path(OUTPUT_DIR) / "last-checkpoint" / "trainer_state.json"
        try:
            step       = json.loads(state_path.read_text()).get("global_step", 0)
            target_dir = Path(OUTPUT_DIR) / f"checkpoint-{step}"
            src_dir    = Path(OUTPUT_DIR) / "last-checkpoint"
            if not target_dir.exists():
                src_dir.rename(target_dir)
                print(f"    Renamed last-checkpoint/ → checkpoint-{step}/")
            else:
                print(f"    checkpoint-{step}/ already exists locally — using it")
        except Exception as e:
            print(f"    Rename failed ({e}) — Trainer will still try to find it")
        return True

    # ── Option 2: scan for checkpoint-N/ (fallback) ──────────────────────
    ckpt_names: set[str] = set()
    for f in all_files:
        part = f.split("/")[0]
        if part.startswith("checkpoint-"):
            try:
                int(part.split("-", 1)[1])
                ckpt_names.add(part)
            except ValueError:
                pass

    if not ckpt_names:
        print("    No checkpoints found — starting fresh")
        return False

    latest = max(ckpt_names, key=lambda x: int(x.split("-", 1)[1]))
    return _download(latest, latest)


HAS_CHECKPOINT = hub_resume()
DATASET_PATH   = "./processed_train"   # Shared between Cell 3 and Cell 4's train()


# ─────────────────────────────────────────────────────────────────────────
#  TOKENIZER  (loaded here for dataset formatting; re-created inside DDP)
# ─────────────────────────────────────────────────────────────────────────

print("\n📦  Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

IM_END_STR = "<|im_end|>"
IM_END_ID  = tokenizer.convert_tokens_to_ids(IM_END_STR)
assert IM_END_ID != tokenizer.unk_token_id, \
    "FATAL: <|im_end|> resolved to UNK — wrong tokenizer or model name!"
print(f"    <|im_end|> token ID : {IM_END_ID}  ✅")

DEFAULT_SYSTEM = (
    "You are Qwen, created by Alibaba Cloud. "
    "You are a helpful and honest assistant."
)

_ROLE = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant",
    "bot": "assistant", "model": "assistant",
    "chatgpt": "assistant", "claude": "assistant",
    "system": "system", "sys": "system",
}


# ─────────────────────────────────────────────────────────────────────────
#  FORMAT PIPELINE
#  Every example → [{role, content}, …] → apply_chat_template → text string
#
#  Qwen3.5-4B chat format (what apply_chat_template produces):
#
#    <|im_start|>system
#    You are Qwen...<|im_end|>
#    <|im_start|>user
#    [question]<|im_end|>
#    <|im_start|>assistant
#    <think>
#    [reasoning chain — this is the most important part to preserve]
#    </think>
#    [final answer]<|im_end|>
# ─────────────────────────────────────────────────────────────────────────

def _to_messages(ex: dict) -> Optional[list]:
    """
    Converts any dataset row format to [{role, content}] for apply_chat_template.
    Tries 6 formats in priority order. Returns None for unusable rows.
    """

    # 1. OpenAI messages format
    raw = ex.get("messages")
    if isinstance(raw, list) and raw:
        msgs = [
            {"role": _ROLE.get(str(m.get("role", "")).lower().strip(), ""),
             "content": str(m.get("content") or m.get("text") or "").strip()}
            for m in raw if isinstance(m, dict)
        ]
        msgs = [m for m in msgs if m["role"] and m["content"]]
        if msgs and any(m["role"] == "assistant" for m in msgs):
            return msgs

    # 2. ShareGPT conversations format  (human/gpt role names)
    convs = ex.get("conversations")
    if isinstance(convs, list) and convs:
        msgs = []
        for t in convs:
            if not isinstance(t, dict):
                continue
            role    = _ROLE.get(str(t.get("from") or t.get("role") or "").lower().strip(), "")
            content = str(t.get("value") or t.get("content") or "").strip()
            if role and content:
                msgs.append({"role": role, "content": content})
        if msgs and any(m["role"] == "assistant" for m in msgs):
            return msgs

    # 3. Prompt / response pairs
    q = str(ex.get("instruction") or ex.get("prompt") or
            ex.get("question")    or ex.get("input")   or "").strip()
    a = str(ex.get("output")   or ex.get("response") or
            ex.get("completion") or ex.get("answer")   or "").strip()
    if q and a:
        sys_ = str(ex.get("system") or "").strip()
        return [
            {"role": "system",    "content": sys_ or DEFAULT_SYSTEM},
            {"role": "user",      "content": q},
            {"role": "assistant", "content": a},
        ]

    # 4. Raw text with a <think>…</think> block  (reasoning traces)
    text = str(ex.get("text") or ex.get("content") or "").strip()
    if text and "<think>" in text and "</think>" in text:
        ctx = str(ex.get("context") or ex.get("query") or "").strip()
        return [
            {"role": "system",    "content": DEFAULT_SYSTEM},
            {"role": "user",      "content": ctx or "Solve the following step by step."},
            {"role": "assistant", "content": text},
        ]

    # 5. Code files  (GitHub, security datasets)
    code = str(ex.get("code") or ex.get("func_code_string") or
               ex.get("whole_func_string") or "").strip()
    lang = str(ex.get("language") or ex.get("programming_language") or "code").lower()
    if code and len(code) > 40:
        return [
            {"role": "system",    "content": DEFAULT_SYSTEM},
            {"role": "user",      "content": f"Write a complete {lang} program."},
            {"role": "assistant", "content": f"```{lang}\n{code}\n```"},
        ]

    # 6. Plain text fallback  (only substantial text)
    if text and len(text) > 200:
        return [
            {"role": "system",    "content": DEFAULT_SYSTEM},
            {"role": "user",      "content": "Continue the following text:"},
            {"role": "assistant", "content": text},
        ]

    return None   # Unusable example


def _format_and_validate(ex: dict) -> dict:
    """
    Full per-example pipeline:
      dict → messages → apply_chat_template → enforce <|im_end|> → {"text": ...}

    Guarantees:
      • Text uses official Qwen3.5 tokens (<|im_start|>, <|im_end|>, <think>)
      • Text ends with <|im_end|>  (the stop signal — critical for preventing loops)
      • Text tokenises to ≥ 20 tokens  (rejects trivial formatting artifacts)

    Returns {"text": ""} for unusable rows; these are filtered out downstream.
    """
    msgs = _to_messages(ex)
    if msgs is None:
        return {"text": ""}

    try:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize              = False,
            add_generation_prompt = False,
            # enable_thinking defaults to True for Qwen3.5 —
            # this preserves the <think>…</think> structure
        )
    except Exception:
        return {"text": ""}

    # Enforce <|im_end|> termination
    text = text.rstrip()
    if not text.endswith(IM_END_STR):
        text = text + IM_END_STR

    # Reject trivially short examples
    if len(tokenizer(text, add_special_tokens=False)["input_ids"]) < 20:
        return {"text": ""}

    return {"text": text}


def _load_source(source, max_rows: int) -> Optional[object]:
    """Load, cap, format, and filter one dataset source."""
    spec_str = str(source)
    try:
        ds = (
            load_dataset(source[0], source[1], split="train", trust_remote_code=True)
            if isinstance(source, tuple)
            else load_dataset(source, split="train", trust_remote_code=True)
        )

        n  = min(max_rows, len(ds))
        ds = ds.select(range(n))

        # batched=True + num_proc=2 → ~40× faster than per-row processing
        def _batch_format(batch: dict) -> dict:
            keys = list(batch.keys())
            n_   = len(batch[keys[0]])
            return {"text": [
                _format_and_validate({k: batch[k][i] for k in keys})["text"]
                for i in range(n_)
            ]}

        ds = ds.map(
            _batch_format,
            batched        = True,
            batch_size     = 128,
            num_proc       = 2,
            remove_columns = ds.column_names,
            desc           = f"  {spec_str[:42]}",
        )

        before = len(ds)
        ds     = ds.filter(lambda x: len(x["text"]) > 0, num_proc=2)
        after  = len(ds)

        print(f"  ✅  {spec_str[:60]:<60}  {after:>6,}  ({before-after} skipped)")
        return ds if after > 0 else None

    except Exception as e:
        print(f"  ⚠️  {spec_str[:60]:<60}  FAILED — {e}")
        return None


print("\n📊  Loading & formatting datasets…")
all_ds = []
for src in DATASET_SOURCES:
    cap = STYLE_CAPS.get(src if isinstance(src, str) else src[0], MAX_ROWS_PER_SOURCE)
    ds  = _load_source(src, cap)
    if ds is not None:
        all_ds.append(ds)

if not all_ds:
    raise RuntimeError("All dataset sources failed. Check HF_TOKEN and source names.")

train_dataset = concatenate_datasets(all_ds).shuffle(seed=SEED)

# ── Token length statistics ───────────────────────────────────────────────
print(f"\n📈  Dataset stats:")
print(f"    Total rows : {len(train_dataset):,}")

_n      = min(1_000, len(train_dataset))
_sample = random.sample(range(len(train_dataset)), _n)
_lens   = sorted([
    len(tokenizer(train_dataset[i]["text"], add_special_tokens=False)["input_ids"])
    for i in _sample
])
_pct_trunc = sum(1 for l in _lens if l > SEQ_LEN) / _n * 100

print(f"    p50={_lens[int(_n*.50)]}  p75={_lens[int(_n*.75)]}  "
      f"p90={_lens[int(_n*.90)]}  p99={_lens[int(_n*.99)]}")
print(f"    ~{_pct_trunc:.1f}% of examples exceed SEQ_LEN={SEQ_LEN} "
      f"→ truncated, <|im_end|> enforced by Qwen35Collator")

# ── Save to disk ──────────────────────────────────────────────────────────
print(f"\n💾  Saving to {DATASET_PATH}…")
train_dataset.save_to_disk(DATASET_PATH)
print(f"    {len(train_dataset):,} rows saved")

del all_ds
gc.collect()
print("\n✅  Cell 3 complete — run Cell 4 to start training.")


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 4 — DDP TRAINING via notebook_launcher
#
#  notebook_launcher forks 2 processes (Linux / Kaggle — fork is safe here).
#  Each process:
#    • Gets its own GPU  (rank 0 → GPU 0, rank 1 → GPU 1)
#    • Loads the FULL Qwen3.5-4B model in FP16 (~8 GB per GPU)
#    • Processes different mini-batches (DistributedSampler splits data)
#    • All-reduces gradients after backward  (DDP synchronisation)
#    • Only rank 0 saves checkpoints + pushes to HF Hub
#
#  Why device_map=None (not "auto"):
#    device_map="auto" = pipeline parallelism: GPU0 → layers 0-15,
#    GPU1 → layers 16-31, processed SEQUENTIALLY with batch=1.
#    GPU1 idles while GPU0 runs and vice versa — ~10-15% speedup only.
#
#    device_map=None + notebook_launcher(num_processes=2) = data parallelism:
#    BOTH GPUs process DIFFERENT sequences SIMULTANEOUSLY → ~1.8× speedup.
#
#  ⚠️  Do NOT load the model before calling notebook_launcher.
#      CUDA contexts cannot be inherited across fork.
# ═══════════════════════════════════════════════════════════════════════════

import contextlib
import math
import time as _time
from typing import Optional   # ← explicit import here (defensive — train() runs in fork)

from accelerate import notebook_launcher
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


# ─────────────────────────────────────────────────────────────────────────
#  Qwen35Collator — enforces <|im_end|> on truncated sequences
#
#  Problem:
#    SFTTrainer truncates examples that exceed SEQ_LEN tokens. After
#    truncation, the last token is whatever was at position SEQ_LEN in
#    the middle of a reasoning chain — often a mid-word subword piece.
#    Training on sequences that end mid-thought teaches the model that
#    reasoning chains don't need to terminate, directly causing the
#    "never stops" / "hallucination loop" failure mode.
#
#  Fix:
#    After the parent collator runs (which handles truncation and label
#    masking), find the last real (non-padding) token in each sequence
#    and replace it with <|im_end|>. The model always sees a stop signal,
#    even if the reasoning chain was cut short by truncation.
#
#  Label handling:
#    The replaced token's label is set to <|im_end|>'s ID (never -100),
#    so the model is always trained to predict the stop signal.
# ─────────────────────────────────────────────────────────────────────────

class Qwen35Collator(DataCollatorForCompletionOnlyLM):

    def __init__(self, im_end_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_end_id = im_end_id

    def __call__(self, features):
        batch  = super().__call__(features)
        ids    = batch["input_ids"]    # (B, T) long tensor
        labels = batch["labels"]       # (B, T) long tensor, -100 for masked

        pad_id = self.tokenizer.pad_token_id or 0

        for i in range(ids.size(0)):
            non_pad = (ids[i] != pad_id).nonzero(as_tuple=False)
            if non_pad.numel() == 0:
                continue
            last = non_pad[-1].item()
            if ids[i, last].item() != self.im_end_id:
                ids[i, last]    = self.im_end_id
                labels[i, last] = self.im_end_id   # Always train on the stop token

        batch["input_ids"] = ids
        batch["labels"]    = labels
        return batch


# ─────────────────────────────────────────────────────────────────────────
#  HubPruneCallback — prevents hub repo from growing unboundedly
#
#  Problem:
#    hub_strategy="checkpoint" pushes a new checkpoint-N/ folder to the
#    hub at every save. With SAVE_EVERY_STEPS=300 and TRAIN_STEPS=30,000
#    that's 100 pushes = ~100 × 100 MB = ~10 GB of checkpoint data on hub.
#
#  Fix:
#    After each successful push, delete all but the 3 newest checkpoint-N/
#    folders from the hub (keeps last-checkpoint/ untouched — that's the
#    primary resume target).
# ─────────────────────────────────────────────────────────────────────────

class HubPruneCallback(TrainerCallback):
    """Deletes old checkpoint-N/ folders from the HF Hub after each save."""

    def __init__(self, hub_repo: str, token: str, keep: int = 3):
        self.repo  = hub_repo
        self.token = token
        self.keep  = keep
        self._api  = HfApi(token=token)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kw):
        if not state.is_world_process_zero:
            return   # Only rank 0 prunes
        try:
            all_files = list(self._api.list_repo_files(self.repo))
        except Exception:
            return

        names: list[str] = sorted(
            {f.split("/")[0] for f in all_files
             if f.split("/")[0].startswith("checkpoint-")},
            key = lambda x: int(x.split("-", 1)[1]),
        )

        to_delete = names[: max(0, len(names) - self.keep)]
        for old in to_delete:
            for f in all_files:
                if f.startswith(f"{old}/"):
                    try:
                        self._api.delete_file(
                            path_in_repo = f,
                            repo_id      = self.repo,
                            token        = self.token,
                        )
                    except Exception:
                        pass
            print(f"\n  🗑️  Pruned old hub checkpoint: {old}")


# ─────────────────────────────────────────────────────────────────────────
#  TimingCallback — prints real sec/step after step 10
# ─────────────────────────────────────────────────────────────────────────

class TimingCallback(TrainerCallback):

    def __init__(self):
        self._times: list[float] = []
        self._t0: Optional[float] = None

    def on_step_begin(self, args, state, control, **kw):
        self._t0 = _time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if self._t0 is not None:
            self._times.append(_time.perf_counter() - self._t0)

        if len(self._times) == 10:
            avg      = sum(self._times) / 10
            left     = args.max_steps - state.global_step
            eta_hrs  = left * avg / 3600
            acct_est = math.ceil(eta_hrs / 30)   # 30 clock-hrs per Kaggle account
            sess_est = math.ceil(eta_hrs / 12)    # 12-hr max per session

            print(f"\n{'═'*62}")
            print(f"  ⏱️  TIMING REPORT  (average of first 10 steps)")
            print(f"  sec / optimiser step     : {avg:.1f}")
            print(f"  steps / hour             : {3600/avg:.0f}")
            print(f"  steps remaining          : {left:,}")
            print(f"  estimated hours left     : {eta_hrs:.1f}")
            print(f"  estimated sessions left  : ~{sess_est}")
            print(f"  Kaggle accounts needed   : ~{acct_est}  (30 hr/acc)")
            print(f"{'═'*62}\n")


# ─────────────────────────────────────────────────────────────────────────
#  SafeSFTTrainer — NaN/Inf protection
#
#  FP16 on T4 can produce NaN/Inf from overflow in long activation sequences.
#  Without detection, one bad batch propagates corrupt weights forward.
#
#  CRITICAL CONTRACT (HF Trainer backward protocol):
#    self.accelerator.backward(loss) receives the FULL undivided loss.
#    Accelerate's internal GradScaler (enabled by fp16=True in SFTConfig)
#    handles FP16 underflow scaling automatically.
#    The returned value is loss / grad_accum for Trainer's LOGGING only.
#
#  Previous versions divided BEFORE backward → gradients were 8× too small
#  → effective LR was 1e-4 / 8 = 1.25e-5 for the ENTIRE run.
# ─────────────────────────────────────────────────────────────────────────

class SafeSFTTrainer(SFTTrainer):

    NAN_LIMIT = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nan_count = 0

    def training_step(self, model, inputs, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        try:
            ctx = self.compute_loss_context_manager()
        except AttributeError:
            ctx = contextlib.nullcontext()

        with ctx:
            loss = self.compute_loss(model, inputs, **kwargs)

        loss_val = loss.detach().float().item()

        if not math.isfinite(loss_val):
            self._nan_count += 1
            if self.accelerator.is_main_process:
                print(f"\n  ⚠️  NaN/Inf loss at step ~{self.state.global_step} "
                      f"({self._nan_count}/{self.NAN_LIMIT}) — skipping step")
            if self._nan_count >= self.NAN_LIMIT:
                raise RuntimeError(
                    f"Training aborted: {self.NAN_LIMIT} NaN/Inf losses in a row.\n"
                    "Likely FP16 overflow. Try lowering LR or SEQ_LEN."
                )
            model.zero_grad()
            return loss.new_tensor(0.0)

        self._nan_count = 0
        self.accelerator.backward(loss)   # ← FULL loss; GradScaler handles FP16 scaling
        return loss.detach() / self.args.gradient_accumulation_steps


# ─────────────────────────────────────────────────────────────────────────
#  train()  — runs inside each DDP process
#
#  All global variables from Cell 2 / Cell 3 (HF_TOKEN, HUB_REPO,
#  SEQ_LEN, TRAIN_STEPS, etc.) are inherited via Linux fork.
#  This is safe on Kaggle (Linux) but would fail on Windows (spawn).
# ─────────────────────────────────────────────────────────────────────────

def train():
    import gc
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig

    # Re-create tokenizer inside the fork (tokenizers use global state)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    im_end_id = tok.convert_tokens_to_ids(IM_END_STR)

    # Load pre-processed dataset from disk (Arrow memory-mapped — no RAM duplication)
    dataset = load_from_disk(DATASET_PATH)

    # ── Model (device_map=None — REQUIRED for DDP) ────────────────────────
    # Trainer/Accelerate moves it to the correct GPU AFTER DDP initialisation.
    # Using device_map="auto" here would create pipeline-parallel placement
    # that is incompatible with DDP's data-parallel gradient all-reduce.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype       = torch.float16,   # FP16: correct for T4 (no native BF16)
        device_map        = None,            # ← critical: let DDP place the model
        use_cache         = False,           # required for gradient checkpointing
        trust_remote_code = True,
    )

    # ── LoRA (passed to SFTTrainer, NOT applied manually) ─────────────────
    # SFTTrainer calls get_peft_model() internally. Calling it here TOO would
    # apply LoRA twice, causing the adapter to be frozen on the second pass.
    peft_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    )

    # ── Collator (token IDs not string — context-safe for special tokens) ─
    response_template_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    collator = Qwen35Collator(
        im_end_id         = im_end_id,
        response_template = response_template_ids,
        tokenizer         = tok,
        mlm               = False,
    )

    # ── SFTConfig ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir = OUTPUT_DIR,

        # SFT fields
        max_seq_length     = SEQ_LEN,
        dataset_text_field = "text",
        packing            = False,   # REQUIRED: GatedDeltaNet cannot use packing

        # Steps
        max_steps                   = TRAIN_STEPS,
        per_device_train_batch_size = BATCH_PER_DEVICE,
        gradient_accumulation_steps = GRAD_ACCUM,

        # LR
        learning_rate     = LR,
        warmup_steps      = WARMUP_STEPS,
        lr_scheduler_type = "cosine",
        weight_decay      = WEIGHT_DECAY,

        # Precision (fp16 = correct for T4 — no native BF16 hardware)
        fp16 = True,
        bf16 = False,

        # Memory
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        # use_reentrant=False: required for DDP — reentrant checkpointing
        # conflicts with DDP's gradient hooks on the same parameters.

        # Optimizer (8-bit Adam: same convergence, ~300 MB less VRAM/GPU)
        optim = "adamw_8bit",

        # Periodic VRAM flush: prevents fragmentation OOM during 12-hr sessions
        torch_empty_cache_steps = 50,

        # Logging
        logging_steps = 10,
        report_to     = "none",

        # Checkpointing
        # save_strategy="steps" is REQUIRED — without it, save_steps is silently
        # ignored and NO checkpoints are ever written to disk.
        save_strategy    = "steps",
        save_steps       = SAVE_EVERY_STEPS,
        save_total_limit = KEEP_CHECKPOINTS,   # local disk only

        # HF Hub push
        # hub_strategy="checkpoint" does two things on every save:
        #   (a) pushes checkpoint-N/ to hub  (like every_save)
        #   (b) updates last-checkpoint/     (for easy resume via hub_resume())
        # hub_always_push=True prevents skipping a push when the previous
        # upload is still in-flight on slow Kaggle connections.
        push_to_hub    = True,
        hub_model_id   = HUB_REPO,
        hub_strategy   = "checkpoint",
        hub_token      = HF_TOKEN,
        hub_always_push = True,

        # Misc
        seed                       = SEED,
        dataloader_pin_memory      = False,
        remove_unused_columns      = True,
        # ddp_find_unused_parameters=False: LoRA only trains adapter params →
        # no unused parameters → DDP doesn't need to scan for them (faster)
        ddp_find_unused_parameters = False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SafeSFTTrainer(
        model         = model,
        train_dataset = dataset,
        peft_config   = peft_config,   # SFTTrainer applies LoRA internally
        tokenizer     = tok,
        args          = training_args,
        data_collator = collator,
        callbacks     = [
            TimingCallback(),
            HubPruneCallback(HUB_REPO, HF_TOKEN, keep=KEEP_HUB_CKPTS),
        ],
    )

    # trainer.accelerator is set up by SFTTrainer — use it directly.
    # Do NOT create a new Accelerator() here — it would conflict with
    # the DDP process group already initialised by the trainer.
    is_main = trainer.accelerator.is_main_process

    if is_main:
        trainer.model.print_trainable_parameters()
        print(f"\n{'═'*70}")
        print(f"  🔥  TRAINING")
        print(f"  Steps    : {TRAIN_STEPS:,}  |  Resume: {HAS_CHECKPOINT}")
        print(f"  SEQ_LEN  : {SEQ_LEN}")
        print(f"  Eff batch: {BATCH_PER_DEVICE} × 2 GPUs × {GRAD_ACCUM} = "
              f"{BATCH_PER_DEVICE * 2 * GRAD_ACCUM} seq/step")
        print(f"  LR       : {LR:.0e}  →  {WARMUP_STEPS} warmup → cosine decay")
        print(f"  Save     : every {SAVE_EVERY_STEPS} steps  →  {HUB_REPO}")
        print(f"{'═'*70}\n")

    trainer.train(resume_from_checkpoint=HAS_CHECKPOINT)

    if is_main:
        print("\n📤  Pushing final checkpoint…")
    trainer.push_to_hub("Final checkpoint — training complete")
    if is_main:
        print(f"\n✅  Done. Model at: https://huggingface.co/{HUB_REPO}")
        print()
        print("  To resume on a new session / next account:")
        print("  → Run Cell 2 (same HF_TOKEN + HUB_REPO)")
        print("  → Run Cell 3  (downloads last-checkpoint/, re-formats dataset)")
        print("  → Run Cell 4  (resumes from last saved step automatically)")

    gc.collect()
    torch.cuda.empty_cache()


# ── Launch ────────────────────────────────────────────────────────────────
from huggingface_hub import HfApi   # needed by HubPruneCallback in forked proc

print("🚀  Launching DDP on 2×T4  (data-parallel, ~1.8× speedup)…\n")

notebook_launcher(
    train,
    num_processes = 2,
    use_port      = "29500",
)
