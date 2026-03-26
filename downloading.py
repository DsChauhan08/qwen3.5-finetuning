import subprocess, sys

def pip(args: str, ignore_errors: bool = False):
    result = subprocess.run(
        [sys.executable, "-m", "pip"] + args.split(),
        capture_output=True, text=True
    )
    if result.returncode != 0 and not ignore_errors:
        print(f"WARN [{args[:40]}]:\n{result.stderr[-300:]}")
    return result.returncode == 0

print("Step 1/4 — Removing stale packages…")
pip("uninstall -y unsloth unsloth_zoo trl transformers tokenizers "
    "datasets peft accelerate bitsandbytes triton", ignore_errors=True)

print("Step 2/4 — Upgrading pip…")
pip("install --upgrade pip -q")

print("Step 3/4 — Installing Unsloth (Kaggle T4 build)…")
pip('install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git" -q')

print("Step 4/4 — Installing supporting packages…")
pip("install "
    '"transformers>=4.51.0" '
    '"datasets>=3.0.0" '
    '"huggingface_hub>=0.27.0" '
    '"peft>=0.14.0" '
    '"trl>=0.15.2" '
    '"accelerate>=1.5.0" '
    '"bitsandbytes>=0.43.0" '
    "-q")

print()
print("=" * 60)
print("✅  INSTALL DONE")
print("   ➜  Now click  Session → Restart Session")
print("   ➜  Then run Cell 2 (Config) + Cell 3 (Training)")
print("=" * 60)
