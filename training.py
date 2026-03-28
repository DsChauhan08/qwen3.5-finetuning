# ═══════════════════════════════════════════════════════════════════════════
#  CELL 3 — PREPARATION  (streaming version — no full dataset downloads)
#
#  WHAT CHANGED vs previous version:
#  ✅  streaming=True on every load_dataset() call
#      → rows fetched on-the-fly from HF Hub, never fully downloaded
#      → nvidia/OpenCodeReasoning-2 (28 GB on disk) becomes ~0 MB on disk
#  ✅  .take(max_rows) replaces .select(range(n))
#      → streaming datasets don't support .select(); .take() is the equivalent
#  ✅  num_proc removed from .map() and .filter()
#      → streaming datasets are single-process iterators; num_proc would crash
#  ✅  trust_remote_code=False (removed entirely)
#      → HF deprecated trust_remote_code for datasets; it warns and ignores it
#  ✅  len(ds) replaced with row counter
#      → streaming datasets have no known length before iteration
#  ✅  Dataset stats use the saved Arrow file (post-save), not the stream
# ═══════════════════════════════════════════════════════════════════════════

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

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.version.cuda}")
print(f"GPUs    : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB")

# ── HF auth ────────────────────────────────────────────────────────────────
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(HUB_REPO, private=True, exist_ok=True)
    print(f"\n✅  HF repo: https://huggingface.co/{HUB_REPO}")
except Exception as e:
    print(f"ℹ️   Repo note: {e}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Resume ─────────────────────────────────────────────────────────────────
def hub_resume() -> bool:
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
                print(f"    ⚠️  trainer_state.json not found — corrupt?")
                return False
            step = json.loads(state_path.read_text()).get("global_step", "?")
            print(f"    ✅  Checkpoint ready at step {step}")
            return True
        except Exception as e:
            print(f"    Download failed: {e}")
            return False

    # Try last-checkpoint/ first (hub_strategy="checkpoint" always maintains this)
    if _download("last-checkpoint", "last-checkpoint/"):
        state_path = Path(OUTPUT_DIR) / "last-checkpoint" / "trainer_state.json"
        try:
            step       = json.loads(state_path.read_text()).get("global_step", 0)
            target_dir = Path(OUTPUT_DIR) / f"checkpoint-{step}"
            src_dir    = Path(OUTPUT_DIR) / "last-checkpoint"
            if not target_dir.exists():
                src_dir.rename(target_dir)
                print(f"    Renamed last-checkpoint/ → checkpoint-{step}/")
        except Exception as e:
            print(f"    Rename note: {e}")
        return True

    # Fallback: scan for checkpoint-N/
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
DATASET_PATH   = "./processed_train"

# ── Tokenizer ──────────────────────────────────────────────────────────────
print("\n📦  Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # no trust_remote_code needed
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
    "human": "user",     "user": "user",
    "gpt": "assistant",  "assistant": "assistant",
    "bot": "assistant",  "model": "assistant",
    "chatgpt": "assistant", "claude": "assistant",
    "system": "system",  "sys": "system",
}

# ─────────────────────────────────────────────────────────────────────────
#  FORMAT PIPELINE  (same as v5 — unchanged)
# ─────────────────────────────────────────────────────────────────────────

def _to_messages(ex: dict) -> Optional[list]:
    # 1. OpenAI messages
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

    # 2. ShareGPT conversations
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

    # 4. Raw text with <think> block
    text = str(ex.get("text") or ex.get("content") or "").strip()
    if text and "<think>" in text and "</think>" in text:
        ctx = str(ex.get("context") or ex.get("query") or "").strip()
        return [
            {"role": "system",    "content": DEFAULT_SYSTEM},
            {"role": "user",      "content": ctx or "Solve the following step by step."},
            {"role": "assistant", "content": text},
        ]

    # 5. Code files
    code = str(ex.get("code") or ex.get("func_code_string") or
               ex.get("whole_func_string") or "").strip()
    lang = str(ex.get("language") or ex.get("programming_language") or "code").lower()
    if code and len(code) > 40:
        return [
            {"role": "system",    "content": DEFAULT_SYSTEM},
            {"role": "user",      "content": f"Write a complete {lang} program."},
            {"role": "assistant", "content": f"```{lang}\n{code}\n```"},
        ]

    # 6. Plain text fallback
    if text and len(text) > 200:
        return [
            {"role": "system",    "content": DEFAULT_SYSTEM},
            {"role": "user",      "content": "Continue the following text:"},
            {"role": "assistant", "content": text},
        ]

    return None


def _format_and_validate(ex: dict) -> dict:
    msgs = _to_messages(ex)
    if msgs is None:
        return {"text": ""}
    try:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize              = False,
            add_generation_prompt = False,
        )
    except Exception:
        return {"text": ""}

    text = text.rstrip()
    if not text.endswith(IM_END_STR):
        text = text + IM_END_STR

    if len(tokenizer(text, add_special_tokens=False)["input_ids"]) < 20:
        return {"text": ""}

    return {"text": text}


# ─────────────────────────────────────────────────────────────────────────
#  STREAMING LOADER
#  Key differences from the old non-streaming version:
#
#  OLD (broken):
#    ds = load_dataset(source, split="train")        # downloads everything
#    ds = ds.select(range(max_rows))                 # then selects rows
#    ds = ds.map(..., num_proc=2)                    # parallel map
#
#  NEW (streaming):
#    ds = load_dataset(source, split="train",
#                      streaming=True)               # no download, iterator only
#    ds = ds.take(max_rows)                          # stop after N rows
#    ds = ds.map(fn)                                 # single-process (required)
#    rows = [r for r in ds if r["text"]]            # materialise into list
#
#  The materialised list is then converted to a HF Dataset for save_to_disk().
#  Total disk usage = only the final processed Arrow file (~30-80 MB/source).
# ─────────────────────────────────────────────────────────────────────────

from datasets import Dataset

def _load_source_streaming(source, max_rows: int) -> Optional[Dataset]:
    """
    Loads a dataset in streaming mode, formats up to max_rows examples,
    and returns a materialised HF Dataset (not an IterableDataset).

    Streaming means: rows are fetched from HF Hub one by one.
    No full parquet files are downloaded to Kaggle's 20 GB disk.
    """
    spec_str = str(source)
    try:
        # streaming=True: returns an IterableDataset — no download happens yet
        if isinstance(source, tuple):
            ds_iter = load_dataset(
                source[0], source[1],
                split   = "train",
                streaming = True,
                # trust_remote_code removed — deprecated for datasets, causes warnings
            )
        else:
            ds_iter = load_dataset(
                source,
                split     = "train",
                streaming = True,
            )

        # .take(N) creates a new IterableDataset that stops after N examples.
        # Only those N rows will be fetched from HF Hub.
        ds_iter = ds_iter.take(max_rows)

        # Apply formatting. No batched=True, no num_proc — streaming datasets
        # are single-process iterators. Batching is irrelevant (we iterate anyway).
        ds_iter = ds_iter.map(_format_and_validate, remove_columns=ds_iter.column_names)

        # Materialise: iterate through the stream and collect valid rows into a list.
        # This is where the actual HTTP requests happen — row by row from HF Hub.
        good_rows: list[str] = []
        skipped = 0
        for row in ds_iter:
            txt = row.get("text", "")
            if txt:
                good_rows.append(txt)
            else:
                skipped += 1

        if not good_rows:
            print(f"  ⚠️  {spec_str[:60]:<60}  0 rows (all skipped)")
            return None

        # Convert list → HF Dataset so it can be concatenated and saved to disk
        result = Dataset.from_dict({"text": good_rows})

        print(f"  ✅  {spec_str[:60]:<60}  {len(result):>6,}  ({skipped} skipped)")
        return result

    except Exception as e:
        print(f"  ⚠️  {spec_str[:60]:<60}  FAILED — {e}")
        return None


print("\n📊  Streaming & formatting datasets…")
print(f"    Max rows per source : {MAX_ROWS_PER_SOURCE:,}")
print(f"    Disk usage          : minimal (streaming — no full downloads)\n")

all_ds = []
for src in DATASET_SOURCES:
    cap = STYLE_CAPS.get(src if isinstance(src, str) else src[0], MAX_ROWS_PER_SOURCE)
    ds  = _load_source_streaming(src, cap)
    if ds is not None:
        all_ds.append(ds)

if not all_ds:
    raise RuntimeError("All dataset sources failed. Check HF_TOKEN and source names.")

train_dataset = concatenate_datasets(all_ds).shuffle(seed=SEED)

# ── Dataset stats ──────────────────────────────────────────────────────────
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
print(f"    ~{_pct_trunc:.1f}% examples exceed SEQ_LEN={SEQ_LEN} "
      f"→ truncated + <|im_end|> enforced by Qwen35Collator")

# ── Save to disk ──────────────────────────────────────────────────────────
print(f"\n💾  Saving processed dataset to {DATASET_PATH}…")
train_dataset.save_to_disk(DATASET_PATH)

import shutil
disk_mb = sum(
    f.stat().st_size for f in Path(DATASET_PATH).rglob("*") if f.is_file()
) / 1e6
print(f"    {len(train_dataset):,} rows saved  ({disk_mb:.0f} MB on disk)")

del all_ds
gc.collect()
print("\n✅  Cell 3 complete — run Cell 4 to start training.")
