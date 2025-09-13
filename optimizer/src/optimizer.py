# optimizer/src/optimizer.py
"""
Stage model weights into a shared /artifacts volume so the server can load fast.
"""

import os
import sys
from huggingface_hub import snapshot_download

MODEL = os.getenv("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ART_OUT = os.getenv("ARTIFACT_MODEL_DIR", "/artifacts/model")


def main() -> None:
    os.makedirs(ART_OUT, exist_ok=True)
    print(f"[optimizer] staging {MODEL} → {ART_OUT}", flush=True)

    # Download/copy model snapshot to the shared folder (no symlinks for portability)
    snapshot_download(
        repo_id=MODEL,
        local_dir=ART_OUT,
        local_dir_use_symlinks=False,
    )

    readme_path = os.path.join(ART_OUT, "STAGED_BY_OPTIMIZER.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"Staged model: {MODEL}\n")

    print("[optimizer] done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[optimizer] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
