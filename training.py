import os, gc, json, queue, random, shutil, threading, time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"]    = "false"
os.environ["WANDB_DISABLED"]            = "true"
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"

import torch
import numpy as np
import datasets

datasets.config.IN_MEMORY_MAX_SIZE = 80 * 1024 * 1024  

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from unsloth import FastModel                         
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import HfApi, login, snapshot_download
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# ── Added New Datasets Here ─────────────────────────────────────────
DATASET_SOURCES = [
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "Crownelius/Opus-4.5-3000x-formatted",
    "Crownelius/Opus-4.6-Reasoning-2100x-formatted",
    "Crownelius/Gemini-3-Pro-Opus-4.5-Kimi-K2.5-13000x-formatted",
    "Crownelius/GLM-5.0-8000x-formatted-fixed",
    "Crownelius/Agentic-SFT-1000x",
    "Roman1111111/gpt-5.4-step-by-step-reasoning",
    "Roman1111111/claude-opus-4.6-10000x",
    "Roman1111111/gemini-3.1-pro-hard-high-reasoning",
    "dalisoft/claude-opus-4.6-high-reasoning-700x",
    "artillerywu/DeepResearch-9K",
    "tandevllc/offsec_redteam_codes",
    "TeichAI/Claude-Opus-Dataclaw-Unredacted",
    "nvidia/OpenCodeReasoning-2",
    "nvidia/Nemotron-SFT-Competitive-Programming-v2",
    "nvidia/Nemotron-Terminal-Synthetic-Tasks",
    "nvidia/Nemotron-RL-ReasoningGym-v1",
]

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(HUB_REPO, private=True, exist_ok=True)
    print(f"HF repo ready: {HUB_REPO}")
except Exception as e:
    print(f"Repo creation note: {e}")

OUTPUT_DIR = Path("./qwen_train_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _snap_step(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except Exception:
        return -1

def _find_latest_snap_name(repo_id: str) -> Optional[str]:
    try:
        files = list(api.list_repo_files(repo_id))
        names = sorted(
            {p.split("/")[1] for p in files
             if p.startswith("snapshots/")
             and len(p.split("/")) > 1
             and p.split("/")[1].startswith("snap_")},
            key=_snap_step,
        )
        return names[-1] if names else None
    except Exception:
        return None

def _download_snapshot(repo_id: str) -> Optional[Path]:
    name = _find_latest_snap_name(repo_id)
    if not name:
        return None
    root = OUTPUT_DIR / "resume"
    root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"snapshots/{name}/*",
            local_dir=str(root),
            token=HF_TOKEN,
        )
        p = root / "snapshots" / name
        return p if p.exists() else None
    except Exception as e:
        print(f"Resume download failed: {e}")
        return None

print("\n🔍  Looking for existing checkpoint on HF Hub…")
resume_dir = _download_snapshot(HUB_REPO)
if resume_dir:
    print(f"✅  Checkpoint found: {resume_dir.name}")
else:
    print("ℹ️   No checkpoint — starting fresh.")

_state: dict = {
    "global_step":    0,
    "micro_step":     0,
    "source_offsets": [0] * len(DATASET_SOURCES),
}

if resume_dir and (resume_dir / "state.json").exists():
    try:
        _state.update(json.loads((resume_dir / "state.json").read_text()))
    except Exception as e:
        print(f"⚠️   state.json load failed: {e}")

global_step    = int(_state["global_step"])
micro_step     = int(_state["micro_step"])
source_offsets = list(_state.get("source_offsets", [0] * len(DATASET_SOURCES)))
if len(source_offsets) != len(DATASET_SOURCES):
    source_offsets = [0] * len(DATASET_SOURCES)

print("\n🚀  Loading model…")

base_model, tokenizer = FastModel.from_pretrained(
    model_name    = MODEL_NAME,
    max_seq_length= SEQ_LEN,
    load_in_4bit  = False,        
    load_in_16bit = True,         
    dtype         = torch.bfloat16,  
    full_finetuning = False,
)

if resume_dir and (resume_dir / "tokenizer").exists():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(resume_dir / "tokenizer"), use_fast=True)
    except Exception as e:
        print(f"Tokenizer checkpoint load failed: {e}")

if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if resume_dir and (resume_dir / "adapter").exists():
    model = PeftModel.from_pretrained(
        base_model,
        str(resume_dir / "adapter"),
        is_trainable=True,
    )
else:
    model = FastModel.get_peft_model(
        base_model,
        r            = LORA_R,
        lora_alpha   = LORA_ALPHA,   
        lora_dropout = LORA_DROPOUT, 
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",  
        random_state = SEED,
        bias = "none",
    )

try:
    model.config.use_cache = False
except Exception:
    pass

trainable = [p for p in model.parameters() if p.requires_grad]

device = torch.device("cuda:0")
autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

optimizer = torch.optim.AdamW(
    trainable,
    lr           = LR,
    weight_decay = 0.01,
    betas        = (0.9, 0.95),
    eps          = 1e-8,
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps  = WARMUP_STEPS,
    num_training_steps= TRAIN_STEPS,
)

if resume_dir and (resume_dir / "optimizer.pt").exists():
    try:
        ckpt = torch.load(resume_dir / "optimizer.pt", map_location="cpu")
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    except Exception as e:
        print(f"⚠️  Optimiser load failed: {e}")

_RESP_HEADER     = "<|im_start|>assistant\n"
_RESP_HEADER_IDS = tokenizer.encode(_RESP_HEADER, add_special_tokens=False)
_IM_END_ID       = tokenizer.convert_tokens_to_ids("<|im_end|>")

def _get_labels(input_ids: list) -> list:
    labels = [-100] * len(input_ids)
    n = len(input_ids)
    k = len(_RESP_HEADER_IDS)
    i = 0
    while i <= n - k:
        if input_ids[i : i + k] == _RESP_HEADER_IDS:
            j = i + k
            while j < n:
                labels[j] = input_ids[j]
                if input_ids[j] == _IM_END_ID:
                    j += 1   
                    break
                j += 1
            i = j   
        else:
            i += 1
    return labels

def _to_messages(ex: dict) -> Optional[list]:
    msgs_raw = ex.get("messages")
    if isinstance(msgs_raw, list) and msgs_raw:
        msgs = [{"role": str(m.get("role", "")),
                 "content": str(m.get("content") or m.get("text") or "")}
                for m in msgs_raw if isinstance(m, dict)]
        if (all(m["role"] in ("system", "user", "assistant") for m in msgs)
                and any(m["role"] == "assistant" for m in msgs)
                and all(m["content"] for m in msgs)):
            return msgs

    convs = ex.get("conversations")
    if isinstance(convs, list) and convs:
        _role = {"human": "user", "gpt": "assistant", "system": "system",
                 "user":  "user", "assistant": "assistant", "bot": "assistant"}
        msgs = []
        for t in convs:
            if not isinstance(t, dict): continue
            role    = _role.get(str(t.get("from") or t.get("role") or ""), "")
            content = str(t.get("value") or t.get("content") or "").strip()
            if role and content:
                msgs.append({"role": role, "content": content})
        if msgs and any(m["role"] == "assistant" for m in msgs):
            return msgs

    question = str(
        ex.get("instruction") or ex.get("prompt") or
        ex.get("question")    or ex.get("input")   or ""
    ).strip()
    answer = str(
        ex.get("output")   or ex.get("response") or
        ex.get("completion") or ex.get("answer") or ""
    ).strip()
    if question and answer:
        msgs = []
        sys_text = str(ex.get("system") or "").strip()
        if sys_text: msgs.append({"role": "system", "content": sys_text})
        msgs.append({"role": "user",      "content": question})
        msgs.append({"role": "assistant", "content": answer})
        return msgs

    text = str(ex.get("text") or ex.get("content") or "").strip()
    if text and "<think>" in text and "</think>" in text:
        ctx = str(ex.get("context") or ex.get("query") or "").strip()
        user_msg = ctx if ctx else "Solve the following problem step by step."
        return [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": text},
        ]

    code = str(ex.get("code") or ex.get("func_code_string") or ex.get("whole_func_string") or "").strip()
    lang = str(ex.get("language") or ex.get("programming_language") or "code").strip().lower()
    if code and len(code) > 20:
        return [
            {"role": "user",      "content": f"Write a {lang} program."},
            {"role": "assistant", "content": f"```{lang}\n{code}\n```"},
        ]

    if text and len(text) > 80:
        return [
            {"role": "user",      "content": "Continue:"},
            {"role": "assistant", "content": text},
        ]
    return None  

def _format_one(ex: dict) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    msgs = _to_messages(ex)
    if msgs is None: return None
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
    except Exception:
        return None

    max_chars = SEQ_LEN * 10
    if len(text) > max_chars: text = text[:max_chars]

    ids = tokenizer(
        text, max_length=SEQ_LEN, truncation=True, padding=False,
        add_special_tokens=False, return_tensors=None,
    )["input_ids"]

    if len(ids) < 8: return None  

    labels = _get_labels(ids)
    if all(lbl == -100 for lbl in labels): return None

    return (torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long))

def _load_stream(spec, skip: int = 0):
    try:
        if isinstance(spec, tuple):
            ds = load_dataset(spec[0], spec[1], split="train", streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(spec, split="train", streaming=True, trust_remote_code=True)

        _KEEP = {
            "text", "content", "prompt", "response", "completion", "messages", "conversations", "system", "instruction",
            "output", "input", "question", "answer", "code", "func_code_string", "whole_func_string", "language",
            "programming_language", "context", "query",
        }
        try:
            cols = ds.column_names
            if isinstance(cols, list):
                drop = [c for c in cols if c.lower() not in _KEEP]
                if drop: ds = ds.remove_columns(drop)
        except Exception: pass

        it = iter(ds)
        for _ in range(skip): next(it, None)
        return it
    except Exception as e:
        return None

class SequenceBatcher:
    def __init__(self, tokenizer, batch_size: int, source_offsets: List[int]):
        self.tokenizer      = tokenizer
        self.batch_size     = batch_size
        self.source_offsets = list(source_offsets)
        self._buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._rr_ptr        = 0   
        self._active: List[list] = []
        for idx, spec in enumerate(DATASET_SOURCES):
            it = _load_stream(spec, skip=self.source_offsets[idx])
            if it is not None: self._active.append([idx, spec, it])

    def _next_valid(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self._active: return None
        max_attempts = len(self._active) * 6
        for _ in range(max_attempts):
            if not self._active: return None
            idx_in_list = self._rr_ptr % len(self._active)
            src_idx, spec, it = self._active[idx_in_list]
            self._rr_ptr += 1  
            try:
                ex = next(it)
                self.source_offsets[src_idx] += 1
                result = _format_one(ex)
                del ex
                if result is not None: return result
            except StopIteration:
                self._active.pop(idx_in_list)
                if self._active: self._rr_ptr = self._rr_ptr % len(self._active)
                else: return None
            except Exception: pass
        return None  

    def next_batch(self) -> Optional[dict]:
        while len(self._buf) < self.batch_size:
            item = self._next_valid()
            if item is None: break
            self._buf.append(item)

        if not self._buf: return None
        items = self._buf[: self.batch_size]
        self._buf = self._buf[self.batch_size :]

        if len(items) == 1:
            ids, lbls = items[0]
            return {
                "input_ids":      ids.unsqueeze(0).to(device),
                "attention_mask": torch.ones_like(ids).unsqueeze(0).to(device),
                "labels":         lbls.unsqueeze(0).to(device),
            }

        max_len = max(x[0].size(0) for x in items)
        pad_id  = self.tokenizer.pad_token_id or 0
        all_ids, all_mask, all_lbls = [], [], []
        for ids, lbls in items:
            pad = max_len - ids.size(0)
            all_ids.append(torch.cat([ids,  ids.new_full((pad,), pad_id)]))
            all_mask.append(torch.cat([torch.ones_like(ids), ids.new_zeros(pad)]))
            all_lbls.append(torch.cat([lbls, lbls.new_full((pad,), -100)]))

        return {
            "input_ids":      torch.stack(all_ids).to(device),
            "attention_mask": torch.stack(all_mask).to(device),
            "labels":         torch.stack(all_lbls).to(device),
        }

    def state_dict(self) -> dict:
        return {"source_offsets": self.source_offsets}

batcher = SequenceBatcher(tokenizer=tokenizer, batch_size=BATCH, source_offsets=source_offsets)

_upload_q:   "queue.Queue[Path]" = queue.Queue(maxsize=2)
_stop_event  = threading.Event()
_state_lock  = threading.Lock()
_last_upload_ts = time.time()

def _prune_hf_snapshots(keep: int = KEEP_SNAPSHOTS) -> None:
    try:
        files = list(api.list_repo_files(HUB_REPO))
        names = sorted(
            {p.split("/")[1] for p in files if p.startswith("snapshots/") and p.split("/")[1].startswith("snap_")},
            key=_snap_step,
        )
        to_delete = names[:-keep] if len(names) > keep else []
        for old_name in to_delete:
            for f in files:
                if f.startswith(f"snapshots/{old_name}/"):
                    try: api.delete_file(path_in_repo=f, repo_id=HUB_REPO, token=HF_TOKEN)
                    except Exception: pass
    except Exception as e: pass

def _save_snapshot(reason: str) -> None:
    snap_name = f"snap_{global_step}"
    snap_dir  = OUTPUT_DIR / "snapshots" / snap_name
    snap_dir.mkdir(parents=True, exist_ok=True)

    with _state_lock:
        snap_state = {
            "reason":      reason,
            "global_step": global_step,
            "micro_step":  micro_step,
            "timestamp":   int(time.time()),
            **batcher.state_dict(),
        }

    try: model.save_pretrained(str(snap_dir / "adapter"))
    except Exception as e: pass
    try: tokenizer.save_pretrained(str(snap_dir / "tokenizer"))
    except Exception: pass
    try:
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, snap_dir / "optimizer.pt")
    except Exception as e: pass

    (snap_dir / "state.json").write_text(json.dumps(snap_state, indent=2))
    gc.collect()
    torch.cuda.empty_cache()

    try: _upload_q.put_nowait(snap_dir)
    except queue.Full: pass

def _upload_worker() -> None:
    while True:
        if _stop_event.is_set() and _upload_q.empty(): break
        try: folder = _upload_q.get(timeout=5)
        except queue.Empty: continue
        try:
            api.upload_folder(repo_id=HUB_REPO, folder_path=str(folder), path_in_repo=f"snapshots/{folder.name}", token=HF_TOKEN)
            _prune_hf_snapshots(keep=KEEP_SNAPSHOTS)
        except Exception as e: pass
        finally: _upload_q.task_done()

_uploader = threading.Thread(target=_upload_worker, daemon=True)
_uploader.start()

_session_start     = time.perf_counter()
_start_global_step = global_step

try:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while global_step < TRAIN_STEPS:
        batch = batcher.next_batch()
        if batch is None: break

        with autocast_ctx:
            out  = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                labels         = batch["labels"],   
            )
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        micro_step += 1

        loss_val = loss.detach().float().item() * GRAD_ACCUM
        del out, loss, batch

        if micro_step >= GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()               
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            micro_step   = 0

            if global_step % 10 == 0:
                elapsed   = time.perf_counter() - _session_start
                done      = max(1, global_step - _start_global_step)
                sps       = elapsed / done                  
                eta_s     = sps * max(0, TRAIN_STEPS - global_step)
                cur_lr    = scheduler.get_last_lr()[0]
                print(f"step {global_step:>6}/{TRAIN_STEPS} | loss {loss_val:.4f} | lr {cur_lr:.2e} | eta {eta_s / 3600:.1f}h")

            if global_step % SAVE_EVERY_STEPS == 0:
                _save_snapshot(f"step_{global_step}")
            
            # 6-Hour Time Based Upload Logic
            if time.time() - _last_upload_ts >= UPLOAD_INTERVAL:
                print("\n⏰ 6 Hours elapsed. Queueing snapshot upload...")
                _save_snapshot("time_based")
                _last_upload_ts = time.time()

            if global_step % 100 == 0:
                gc.collect()

    if micro_step > 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        micro_step   = 0

finally:
    print("\n📦  Saving final snapshot…")
    _save_snapshot("final")
    try: _upload_q.join()
    except Exception: pass
    _stop_event.set()
    try: _uploader.join(timeout=90)
    except Exception: pass
    print(f"✅  DONE  —  completed at step {global_step}")
