import os
import json
import re
from pathlib import Path
from typing import Dict, Set, List

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


# -----------------------------
# Config
# -----------------------------
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_DIR = os.path.expanduser("~/datasets/test_data/Set5_512_LR_x4")
OUT_DIR = os.path.expanduser("~/datasets/test_data/captions_texture_Set5_512_LR_x4")

NUM_CHUNKS = 1  # adjust if needed (higher = smaller chunks)

MAX_NEW_TOKENS = 80
REWRITE_MAX_NEW_TOKENS = 50
SEED = 0

INSTRUCTION = (
    "You are a vision expert for super-resolution.\n"
    "Describe ONLY low-level visual properties of this image:\n"
    "- texture (fine/coarse/grainy/smooth)\n"
    "- edges (sharp/soft/jagged), lines, contours\n"
    "- patterns (repetitive/irregular), high-frequency detail\n"
    "- local contrast, noise, blur\n"
    "DO NOT mention objects, people, animals, places, or scene semantics.\n"
    "Start directly with texture or edge descriptors.\n"
    "Return ONE short sentence.\n"
)

# -----------------------------
# Semantic leakage detection
# -----------------------------
SEMANTIC_RE = re.compile(
    r"\b("
    r"road|street|building|house|car|train|tree|plant|grass|sky|water|pool|"
    r"table|food|salad|meat|person|people|man|woman|girl|boy|"
    r"animal|dog|cat|butterfly|tiger|statue|wall|floor|brick|stone|"
    r"rug|carpet|graffiti"
    r")\b",
    re.IGNORECASE,
)


# -----------------------------
# Attribute helpers
# -----------------------------
def needs_rewrite(text: str) -> bool:
    return SEMANTIC_RE.search(text) is not None


def count_attributes(text: str) -> int:
    keywords = [
        "fine", "coarse", "grain", "smooth",
        "sharp", "soft", "jagged",
        "edge", "edges", "line", "lines", "contour",
        "pattern", "patterns",
        "frequency", "detail",
        "contrast", "noise", "blur",
    ]
    t = text.lower()
    return sum(1 for k in keywords if k in t)


def extract_attributes(text: str) -> List[str]:
    t = text.lower()
    attrs = []

    if "fine" in t:
        attrs.append("fine-grained texture")
    if "coarse" in t:
        attrs.append("coarse texture")
    if "grain" in t:
        attrs.append("grainy texture")
    if "smooth" in t:
        attrs.append("smooth surface")
    if "sharp" in t:
        attrs.append("sharp edges")
    if "soft" in t:
        attrs.append("soft edges")
    if "jagged" in t:
        attrs.append("jagged edges")
    if "pattern" in t:
        attrs.append("repetitive patterns")
    if "high-frequency" in t or "high frequency" in t:
        attrs.append("high-frequency detail")
    if "contrast" in t:
        attrs.append("local contrast")
    if "noise" in t:
        attrs.append("visible noise")
    if "blur" in t:
        attrs.append("blur")

    return attrs


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
        print("✅ All chunks already processed.")
        return

    out_path = out_dir / f"captions_chunk_{chunk_id}.jsonl"

    print(f"Device: {device}")
    print(f"Images total: {len(files)}")
    print(f"Processing chunk {chunk_id + 1}/{NUM_CHUNKS}")
    print(f"Output: {out_path}")

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
                images=image,
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
            clean = text

            if needs_rewrite(clean) or count_attributes(clean) < 2:
                attrs = extract_attributes(text)
                if len(attrs) < 2:
                    attrs.append("moderate local contrast")
                if len(attrs) < 2:
                    attrs.append("subtle high-frequency detail")
                clean = ", ".join(attrs[:3]).capitalize() + "."

            rec: Dict[str, str] = {
                "file": img_path.name,
                "caption_raw": text,
                "caption_clean": clean,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Finished chunk {chunk_id}")


if __name__ == "__main__":
    main()
