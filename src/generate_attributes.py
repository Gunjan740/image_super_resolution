import os
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Config (EDIT PER RUN)
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

IMAGE_DIR = os.path.expanduser("~/datasets/DF2K/sample_10_df2klr")
OUT_JSONL = os.path.expanduser("~/datasets/DF2K/sample_10_df2klr_attributes.jsonl")

DATASET_NAME = "DF2K"
SPLIT_NAME = "train"
RESOLUTION = "LR"

MAX_NEW_TOKENS = 200
SEED = 0

# Prompt (fixed)
PROMPT = """USER: <image>
You are a vision expert for image super-resolution.

Given the image, output a JSON object with EXACTLY the following keys and allowed values only:

texture_scale: one of [fine, coarse, mixed]
edge_quality: one of [sharp, soft, blurred]
frequency_detail: one of [high, medium, low]
pattern_structure: one of [repetitive, irregular, none]
contrast_level: one of [high, medium, low]
noise_level: one of [low, medium, high]

Rules:
- Output ONLY valid JSON.
- Use ONLY the allowed values.
- Do NOT mention objects, animals, people, or scenes.
- Do NOT add explanations or extra text.
- Do NOT invent new keys.

Return JSON only.
ASSISTANT:
"""

def main():
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    img_dir = Path(IMAGE_DIR)
    out_path = Path(OUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )

    if not files:
        raise RuntimeError(f"No images found in {img_dir}")

    print("=" * 60, flush=True)
    print(" Super-Resolution Attribute Extraction", flush=True)
    print("=" * 60, flush=True)
    print(f"Dataset     : {DATASET_NAME}", flush=True)
    print(f"Split       : {SPLIT_NAME}", flush=True)
    print(f"Resolution  : {RESOLUTION}", flush=True)
    print(f"Image dir   : {img_dir}", flush=True)
    print(f"Num images  : {len(files)}", flush=True)
    print(f"Device      : {device}", flush=True)
    print("=" * 60, flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, img_path in enumerate(files, start=1):
            image = Image.open(img_path).convert("RGB")

            print(f"[DEBUG] Starting generation for {img_path.name}", flush=True)

            inputs = processor(
                text=PROMPT,
                images=[image],   # IMPORTANT: list input for LLaVA
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

            text = processor.decode(output_ids[0], skip_special_tokens=True)
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:", 1)[-1].strip()

            # Sanity: raw model output
            if idx <= 3:
                print("----- RAW MODEL OUTPUT -----", flush=True)
                print(text, flush=True)
                print("----------------------------", flush=True)

            # Robust JSON extraction
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                print(f"[WARN] No JSON found -> {img_path.name}", flush=True)
                continue

            # FIX: unescape LaTeX-style underscores
            json_text = match.group().replace("\\_", "_")

            try:
                attrs = json.loads(json_text)
            except json.JSONDecodeError:
                print(f"[WARN] JSON parse failed -> {img_path.name}", flush=True)
                continue

            record = {
                "file": img_path.name,
                "dataset": DATASET_NAME,
                "split": SPLIT_NAME,
                "resolution": RESOLUTION,
                **attrs,
            }

            fout.write(json.dumps(record) + "\n")
            fout.flush()
            written += 1

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(files)} | written: {written}", flush=True)

    print("=" * 60, flush=True)
    print(f"Finished writing: {out_path}", flush=True)
    print(f"Total records written: {written}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
