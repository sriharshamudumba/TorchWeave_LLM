import os, json, tarfile, tempfile, uuid

ART_DIR = os.getenv("ART_DIR","/artifacts")

def pack_artifact(model, tokenizer, meta: dict):
    os.makedirs(ART_DIR, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]
    tmp = tempfile.mkdtemp()
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    with open(os.path.join(tmp,"torchweave_meta.json"),"w") as f:
        json.dump(meta, f, indent=2)
    tar_path = os.path.join(ART_DIR, f"model_{run_id}.tar")
    with tarfile.open(tar_path,"w") as tar:
        tar.add(tmp, arcname=".")
    return f"file://{tar_path}"
