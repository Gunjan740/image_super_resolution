import json
from pathlib import Path

CHUNK_DIR = Path("~/datasets/DF2K/captions_chunks_LR_x4_semantic").expanduser()
OUT_FILE = Path("~/datasets/DF2K/df2k_LR_x4_semantic_captions.jsonl").expanduser()

records = {}

for chunk in sorted(CHUNK_DIR.glob("captions_chunk_*.jsonl")):
    with chunk.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[rec["file"]] = rec  # deduplicate by filename

with OUT_FILE.open("w", encoding="utf-8") as f:
    for key in sorted(records):
        f.write(json.dumps(records[key], ensure_ascii=False) + "\n")

print(f" Merged {len(records)} records into {OUT_FILE}")
