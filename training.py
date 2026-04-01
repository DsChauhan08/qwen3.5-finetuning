
# ═══════════════════════════════════════════════════════════════════════════
#  CELL 3 — PREPARATION  (streaming — no full downloads, ~50–150 MB disk)
# ═══════════════════════════════════════════════════════════════════════════

import gc
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
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

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(HUB_REPO, private=True, exist_ok=True)
    print(f"\n✅  HF repo: https://huggingface.co/{HUB_REPO}")
except Exception as e:
    print(f"ℹ️   Repo: {e}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Resume ─────────────────────────────────────────────────────────────────
def hub_resume() -> bool:
    print("\n🔍  Checking HF Hub for checkpoint…")
    try:
        all_files = list(api.list_repo_files(HUB_REPO))
    except Exception as e:
        print(f"    Unreachable ({e}) — fresh start"); return False
    if not all_files:
        print("    Empty repo — fresh start"); return False

    def _dl(folder: str, label: str) -> bool:
        if not any(f.startswith(f"{folder}/") for f in all_files):
            return False
        print(f"    Downloading {label}…")
        try:
            snapshot_download(
                repo_id=HUB_REPO, allow_patterns=[f"{folder}/*"],
                local_dir=OUTPUT_DIR, token=HF_TOKEN,
            )
            sp = Path(OUTPUT_DIR) / folder / "trainer_state.json"
            if not sp.exists():
                print("    ⚠️  trainer_state.json missing"); return False
            step = json.loads(sp.read_text()).get("global_step", "?")
            print(f"    ✅  Ready at step {step}"); return True
        except Exception as e:
            print(f"    Failed: {e}"); return False

    if _dl("last-checkpoint", "last-checkpoint/"):
        sp = Path(OUTPUT_DIR) / "last-checkpoint" / "trainer_state.json"
        try:
            step = json.loads(sp.read_text()).get("global_step", 0)
            tgt  = Path(OUTPUT_DIR) / f"checkpoint-{step}"
            src  = Path(OUTPUT_DIR) / "last-checkpoint"
            if not tgt.exists():
                src.rename(tgt)
                print(f"    Renamed → checkpoint-{step}/")
        except Exception:
            pass
        return True

    ckpts = {f.split("/")[0] for f in all_files if f.split("/")[0].startswith("checkpoint-")}
    if not ckpts:
        print("    No checkpoints — fresh start"); return False
    latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
    return _dl(latest, latest)


HAS_CHECKPOINT = hub_resume()
DATASET_PATH   = "./processed_train"

# ── Tokenizer ──────────────────────────────────────────────────────────────
print("\n📦  Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

IM_END_STR = "<|im_end|>"
IM_END_ID  = tokenizer.convert_tokens_to_ids(IM_END_STR)
assert IM_END_ID != tokenizer.unk_token_id, "FATAL: <|im_end|> is UNK!"
print(f"    <|im_end|> token ID : {IM_END_ID}  ✅")

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful and honest assistant."
_ROLE = {
    "human": "user",      "user": "user",
    "gpt": "assistant",   "assistant": "assistant",
    "bot": "assistant",   "model": "assistant",
    "chatgpt": "assistant", "claude": "assistant",
    "system": "system",   "sys": "system",
}

# ── Format pipeline ────────────────────────────────────────────────────────
def _to_messages(ex: dict) -> Optional[list]:
    for tf in ("text", "content"):
        raw = str(ex.get(tf) or "").strip()
        if raw and "<|im_start|>" in raw and "<|im_end|>" in raw:
            return [{"role": "__RAW__", "content": raw}]

    raw = ex.get("messages")
    if isinstance(raw, list) and raw:
        msgs = []
        for m in raw:
            if not isinstance(m, dict): continue
            role    = _ROLE.get(str(m.get("role", "")).lower().strip(), "")
            content = str(m.get("content") or m.get("text") or "").strip()
            if role and content:
                msgs.append({"role": role, "content": content})
        if msgs and any(m["role"] == "assistant" for m in msgs):
            return msgs

    convs = ex.get("conversations")
    if isinstance(convs, list) and convs:
        msgs = []
        for t in convs:
            if not isinstance(t, dict): continue
            role    = _ROLE.get(str(t.get("from") or t.get("role") or "").lower().strip(), "")
            content = str(t.get("value") or t.get("content") or "").strip()
            if role and content:
                msgs.append({"role": role, "content": content})
        if msgs and any(m["role"] == "assistant" for m in msgs):
            return msgs

    q = str(ex.get("instruction") or ex.get("prompt") or ex.get("question") or ex.get("input") or ex.get("original_input") or "").strip()
    thoughts = str(ex.get("r1_generation") or ex.get("model_thoughts") or "").strip()
    a = str(ex.get("output") or ex.get("response") or ex.get("completion") or ex.get("answer") or ex.get("solution") or ex.get("model_response") or "").strip()
    if q and a:
        sys_ = str(ex.get("system") or "").strip()
        if thoughts: a = f"<think>\n{thoughts}\n</think>\n{a}"
        return [
            {"role": "system",    "content": sys_ or DEFAULT_SYSTEM},
            {"role": "user",      "content": q},
            {"role": "assistant", "content": a},
        ]

    problem  = str(ex.get("problem")  or "").strip()
    thinking = str(ex.get("thinking") or "").strip()
    solution = str(ex.get("solution") or "").strip()
    if problem and (thinking or solution):
        if thinking and solution: asst = f"<think>\n{thinking}\n</think>\n{solution}"
        elif thinking: asst = f"<think>\n{thinking}\n</think>"
        else: asst = solution
        return [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": problem}, {"role": "assistant", "content": asst}]

    query = str(ex.get("query") or ex.get("topic") or ex.get("question") or "").strip()
    traj = str(ex.get("search trajectory") or "").strip()
    report = str(ex.get("report") or ex.get("research") or ex.get("article") or ex.get("final answer") or "").strip()
    if query and report:
        if traj: report = f"<think>\n{traj}\n</think>\n{report}"
        return [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": query}, {"role": "assistant", "content": report}]

    text = str(ex.get("text") or ex.get("content") or "").strip()
    if text and "<think>" in text and "</think>" in text:
        ctx = str(ex.get("context") or ex.get("query") or "").strip()
        return [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": ctx or "Solve the following step by step."}, {"role": "assistant", "content": text}]

    code = str(ex.get("code") or ex.get("func_code_string") or ex.get("whole_func_string") or "").strip()
    lang = str(ex.get("language") or ex.get("programming_language") or "code").lower()
    if code and len(code) > 40:
        return [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": f"Write a complete {lang} program."}, {"role": "assistant", "content": f"```{lang}\n{code}\n```"}]

    if text and len(text) > 200:
        return [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": "Continue the following text:"}, {"role": "assistant", "content": text}]

    return None

def _format_and_validate(ex: dict) -> str:
    try:
        msgs = _to_messages(ex)
        if msgs is None: return ""

        if msgs[0]["role"] == "__RAW__":
            text = msgs[0]["content"].strip()
            if not text.endswith(IM_END_STR): text += IM_END_STR
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            return text if len(ids) >= 20 else ""

        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        text = text.rstrip()
        if not text.endswith(IM_END_STR): text += IM_END_STR
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) >= 20 else ""
    except Exception:
        return ""

def _spec_label(spec) -> str:
    if isinstance(spec, str): return spec
    n = spec.get("name", "?")
    c = spec.get("config", "")
    s = spec.get("split", "train")
    label = f"{n}/{c}" if c else n
    if s and s != "train": label += f"[{s}]"
    return label

def _load_source_streaming(spec, max_rows: int) -> Optional[Dataset]:
    label = _spec_label(spec)
    pad   = 62

    try:
        if isinstance(spec, str):
            ds_iter = load_dataset(spec, split="train", streaming=True, trust_remote_code=True)
        else:
            name   = spec["name"]
            config = spec.get("config")
            split  = spec.get("split", "train")
            if config: ds_iter = load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)
            else: ds_iter = load_dataset(name, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠️  {label:<{pad}}  LOAD FAILED — {str(e)[:80]}")
        return None

    ds_iter = ds_iter.take(max_rows)
    good_rows = []
    skipped = 0
    it = iter(ds_iter)

    while True:
        try: row = next(it)
        except StopIteration: break
        except Exception: skipped += 1; continue

        try:
            txt = _format_and_validate(row)
            if txt: good_rows.append(txt)
            else: skipped += 1
        except Exception:
            skipped += 1

    if not good_rows:
        print(f"  ⚠️  {label:<{pad}}  0 rows (all {skipped} skipped)")
        return None

    result = Dataset.from_dict({"text": good_rows})
    print(f"  ✅  {label:<{pad}}  {len(result):>6,}  ({skipped} skipped)")
    return result


print("\n📊  Streaming & formatting datasets…")
all_ds = []
for spec in DATASET_SOURCES:
    ds_name = spec if isinstance(spec, str) else spec.get("name", "")
    cap     = STYLE_CAPS.get(ds_name, MAX_ROWS_PER_SOURCE)
    ds      = _load_source_streaming(spec, cap)
    if ds is not None:
        all_ds.append(ds)

if not all_ds:
    raise RuntimeError("Every dataset source failed. Check HF_TOKEN and source names.")

train_dataset = concatenate_datasets(all_ds).shuffle(seed=SEED)

print(f"\n📈  Dataset stats:")
print(f"    Total rows : {len(train_dataset):,}")
_n      = min(1_000, len(train_dataset))
_sample = random.sample(range(len(train_dataset)), _n)
_lens   = sorted([len(tokenizer(train_dataset[i]["text"], add_special_tokens=False)["input_ids"]) for i in _sample])
_pct = sum(1 for l in _lens if l > SEQ_LEN) / _n * 100
print(f"    p50={_lens[int(_n*.50)]}  p75={_lens[int(_n*.75)]}  p90={_lens[int(_n*.90)]}  p99={_lens[int(_n*.99)]}")
print(f"    ~{_pct:.1f}% exceed SEQ_LEN={SEQ_LEN} (truncated + <|im_end|> enforced)")

print(f"\n💾  Saving to {DATASET_PATH}…")
train_dataset.save_to_disk(DATASET_PATH)
disk_mb = sum(f.stat().st_size for f in Path(DATASET_PATH).rglob("*") if f.is_file()) / 1e6
print(f"    {len(train_dataset):,} rows saved  ({disk_mb:.0f} MB on disk)")

del all_ds
gc.collect()
print("\n✅  Cell 3 done — run Cell 4.")


# ═══════════════════════════════════════════════════════════════════════════
#  CELL 4 — DDP TRAINING via notebook_launcher
# ═══════════════════════════════════════════════════════════════════════════

import contextlib
import math
import time as _time
from typing import Optional

from accelerate import notebook_launcher
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

# ── Safe TRL Imports (Top level) ───────────────────────────────────────────
from trl import SFTConfig, SFTTrainer
try:
    # older/newer TRL variants
    from trl import DataCollatorForCompletionOnlyLM
except Exception:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    except Exception:
        from trl.data_utils import DataCollatorForCompletionOnlyLM


# ── Collator ──────────────────────────────────────────────────────────────
class Qwen35Collator(DataCollatorForCompletionOnlyLM):
    def __init__(self, im_end_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_end_id = im_end_id

    def __call__(self, features):
        batch  = super().__call__(features)
        ids    = batch["input_ids"]
        labels = batch["labels"]
        pad_id = self.tokenizer.pad_token_id or 0
        for i in range(ids.size(0)):
            non_pad = (ids[i] != pad_id).nonzero(as_tuple=False)
            if non_pad.numel() == 0: continue
            last = non_pad[-1].item()
            if ids[i, last].item() != self.im_end_id:
                ids[i, last]    = self.im_end_id
                labels[i, last] = self.im_end_id
        batch["input_ids"] = ids
        batch["labels"]    = labels
        return batch

# ── Hub prune callback ────────────────────────────────────────────────────
class HubPruneCallback(TrainerCallback):
    def __init__(self, repo: str, token: str, keep: int = 3):
        self.repo = repo; self.token = token; self.keep = keep
        self._api = HfApi(token=token)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kw):
        if not state.is_world_process_zero: return
        try:
            files = list(self._api.list_repo_files(self.repo))
            names = sorted(
                {f.split("/")[0] for f in files if f.split("/")[0].startswith("checkpoint-")},
                key=lambda x: int(x.split("-")[1]),
            )
            for old in names[: max(0, len(names) - self.keep)]:
                for f in files:
                    if f.startswith(f"{old}/"):
                        try: self._api.delete_file(path_in_repo=f, repo_id=self.repo, token=self.token)
                        except Exception: pass
                print(f"\n  🗑️  Pruned: {old}")
        except Exception: pass

# ── Timing callback ───────────────────────────────────────────────────────
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
            avg   = sum(self._times) / 10
            left  = args.max_steps - state.global_step
            eta_h = left * avg / 3600
            n_acc = math.ceil(eta_h / 30)
            print(f"\n{'═'*60}")
            print(f"  ⏱️  TIMING (first 10 steps)")
            print(f"  {avg:.1f} sec/step  →  {3600/avg:.0f} steps/hr")
            print(f"  {left:,} steps left  ≈  {eta_h:.1f} hrs  ≈  {n_acc} accounts")
            print(f"{'═'*60}\n")

# ── NaN-safe trainer ──────────────────────────────────────────────────────
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
                print(f"\n  ⚠️  NaN/Inf at step ~{self.state.global_step} "
                      f"({self._nan_count}/{self.NAN_LIMIT})")
            if self._nan_count >= self.NAN_LIMIT:
                raise RuntimeError("Training aborted: too many NaN losses.")
            model.zero_grad()
            return loss.new_tensor(0.0)

        self._nan_count = 0
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps


# ── train() — runs inside each DDP worker process ─────────────────────────
def train():
    import gc, torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # ── Safe TRL Imports (Inside fork) ──────────────────────────────────
    from trl import SFTConfig, SFTTrainer
    try:
        # older/newer TRL variants
        from trl import DataCollatorForCompletionOnlyLM
    except Exception:
        try:
            from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        except Exception:
            from trl.data_utils import DataCollatorForCompletionOnlyLM

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    im_end_id = tok.convert_tokens_to_ids(IM_END_STR)

    dataset = load_from_disk(DATASET_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype         = torch.float16,
        device_map          = None,
        use_cache           = False,
        attn_implementation = "sdpa",
    )

    peft_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    )

    response_template_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    collator = Qwen35Collator(
        im_end_id=im_end_id, response_template=response_template_ids,
        tokenizer=tok, mlm=False,
    )

    training_args = SFTConfig(
        output_dir = OUTPUT_DIR,
        max_seq_length     = SEQ_LEN,
        dataset_text_field = "text",
        packing            = False,
        max_steps                   = TRAIN_STEPS,
        per_device_train_batch_size = BATCH_PER_DEVICE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate     = LR,
        warmup_steps      = WARMUP_STEPS,
        lr_scheduler_type = "cosine",
        weight_decay      = WEIGHT_DECAY,
        fp16 = True,
        bf16 = False,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        optim = "adamw_8bit",
        torch_empty_cache_steps = 50,
        logging_steps = 10,
        report_to     = "none",
        save_strategy    = "steps",
        save_steps       = SAVE_EVERY_STEPS,
        save_total_limit = KEEP_CHECKPOINTS,
        push_to_hub     = True,
        hub_model_id    = HUB_REPO,
        hub_strategy    = "checkpoint",
        hub_token       = HF_TOKEN,
        hub_always_push = True,
        seed                       = SEED,
        dataloader_pin_memory      = False,
        remove_unused_columns      = True,
        ddp_find_unused_parameters = False,
    )

    trainer = SafeSFTTrainer(
        model         = model,
        train_dataset = dataset,
        peft_config   = peft_config,
        tokenizer     = tok,
        args          = training_args,
        data_collator = collator,
        callbacks     = [
            TimingCallback(),
            HubPruneCallback(HUB_REPO, HF_TOKEN, keep=KEEP_HUB_CKPTS),
        ],
    )

    is_main = trainer.accelerator.is_main_process

    if is_main:
        trainer.model.print_trainable_parameters()
        print(f"\n{'═'*70}")
        print(f"  🔥  TRAINING")
        print(f"  Steps    : {TRAIN_STEPS:,}  |  Resume: {HAS_CHECKPOINT}")
        print(f"  SEQ_LEN  : {SEQ_LEN}  |  attn: sdpa")
        print(f"  Eff batch: 1 × 2 GPUs × {GRAD_ACCUM} accum = {2*GRAD_ACCUM}")
        print(f"  LR       : {LR:.0e}  →  {WARMUP_STEPS} warmup → cosine")
        print(f"  Saves    : every {SAVE_EVERY_STEPS} steps → {HUB_REPO}")
        print(f"{'═'*70}\n")

    trainer.train(resume_from_checkpoint=HAS_CHECKPOINT)

    if is_main:
        print("\n📤  Pushing final checkpoint…")
    trainer.push_to_hub("Final checkpoint — training complete")
    if is_main:
        print(f"\n✅  Done. https://huggingface.co/{HUB_REPO}")
        print("\n  Next session / next account:")
        print("  → Same HF_TOKEN + HUB_REPO in Cell 2")
        print("  → Run Cell 3 (re-streams + downloads last-checkpoint/)")
        print("  → Run Cell 4 (resumes from last saved step)")

    gc.collect()
    torch.cuda.empty_cache()

# ── Launch ────────────────────────────────────────────────────────────────
print("🚀  Launching DDP on 2×T4  (data-parallel — ~1.8× vs single GPU)…\n")
notebook_launcher(train, num_processes=2, use_port="29500")
