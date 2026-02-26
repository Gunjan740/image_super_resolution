import os
import json
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


# -----------------------------
# Config
# -----------------------------
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

IMAGE_DIR = os.path.expanduser("~/datasets/test_data/Set14_512_LR_x4")
OUT_DIR = os.path.expanduser("~/datasets/test_data/captions_chunks_Set14_512_LR_x4_semantic")

NUM_CHUNKS = 1  # adjust if needed (higher = smaller chunks)

MAX_NEW_TOKENS = 80
SEED = 0


# -----------------------------
# Semantic caption instruction
# -----------------------------
INSTRUCTION = (
    "Describe the image in one short, factual sentence.\n"
    "Focus on the main objects and the overall scene.\n"
    "Avoid stylistic language or speculation.\n"
)


# -----------------------------
# Chunk management
# -----------------------------
def get_next_chunk_id(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        int(p.stem.split("_")[-1])
        for p in out_dir.glob("captions_chunk_*.jsonl")
        if p.stem.split("_")[-1].isdigit()
    }
    for i in range(NUM_CHUNKS):
        if i not in existing:
            return i
    return -1  # all done


# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    img_dir = Path(IMAGE_DIR).expanduser()
    out_dir = Path(OUT_DIR).expanduser()

    files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )

    chunk_id = get_next_chunk_id(out_dir)
    if chunk_id == -1:
        print("All chunks already processed.")
        return

    out_path = out_dir / f"captions_chunk_{chunk_id}.jsonl"

    print(f"Device: {device}", flush=True)
    print(f"Images total: {len(files)}", flush=True)
    print(f"Processing chunk {chunk_id + 1}/{NUM_CHUNKS}", flush=True)
    print(f"Output: {out_path}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, img_path in enumerate(files):
            if idx % NUM_CHUNKS != chunk_id:
                continue

            image = Image.open(img_path).convert("RGB")

            prompt = f"USER: <image>\n{INSTRUCTION}\nASSISTANT:"

            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            raw = processor.decode(output_ids[0], skip_special_tokens=True)
            text = raw.split("ASSISTANT:", 1)[-1].strip() if "ASSISTANT:" in raw else raw.strip()

            # For semantic captions: raw == clean
            rec: Dict[str, str] = {
                "file": img_path.name,
                "caption_raw": text,
                "caption_clean": text,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Finished chunk {chunk_id}", flush=True)


if __name__ == "__main__":
    main()
