import pandas as pd
import os
import csv
import random
import subprocess
from PIL import Image, ImageOps

# ── CONFIG ────────────────────────────────────────────────
PARQUET_FILE  = "train.parquet"
TEXT_COLUMN   = "text"
OUTPUT_DIR    = "data/clean"
LABELS_FILE   = "data/labels.csv"
MIN_WORDS     = 2                   # minimum words per image
MAX_WORDS     = 5                   # maximum words per image
TARGET_COUNT  = 3000
FONT_SIZE     = 24
PADDING       = 10
FONTS         = [
    "Noto Sans Devanagari",
    "Noto Serif Devanagari",
    "Lohit Devanagari",
]
# ─────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── STEP 1: Load & clean texts ────────────────────────────
print("Loading parquet...")
df = pd.read_parquet(PARQUET_FILE)
texts = df[TEXT_COLUMN].dropna().astype(str).tolist()
print(f"Loaded {len(texts)} rows")

# ── STEP 2: Split by words with min/max limit ─────────────
def split_text(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    words  = text.strip().split()
    chunks = []
    i      = 0
    while i < len(words):
        # randomly pick chunk size between min and max
        size  = random.randint(min_words, max_words)
        chunk = " ".join(words[i:i+size])
        if len(chunk.strip()) > 2:
            chunks.append(chunk)
        i += size
    return chunks

chunked = [chunk for t in texts for chunk in split_text(t)]
# Filter chunks that are too short
chunked = [t for t in chunked if len(t.strip().split()) >= MIN_WORDS]
print(f"Total chunks after splitting: {len(chunked)}")

# ── STEP 3: Limit & shuffle ───────────────────────────────
random.shuffle(chunked)
chunked = chunked[:TARGET_COUNT]
print(f"Using {len(chunked)} texts for generation")

# ── STEP 4: Distribute across fonts ──────────────────────
per_font     = len(chunked) // len(FONTS)
font_batches = []
for i, font in enumerate(FONTS):
    start = i * per_font
    end   = start + per_font if i < len(FONTS) - 1 else len(chunked)
    font_batches.append((font, chunked[start:end]))

# ── STEP 5: Crop and pad ──────────────────────────────────
def crop_and_pad(img_path, padding=PADDING):
    img      = Image.open(img_path).convert("RGB")
    inverted = ImageOps.invert(img)
    bbox     = inverted.getbbox()

    if bbox:
        img = img.crop(bbox)

    img = ImageOps.expand(img, border=padding, fill="white")
    img.save(img_path)

# ── STEP 6: Generate images using text2image ──────────────
def generate_with_text2image(text, font, output_base, font_size=FONT_SIZE):
    tmp_txt = "tmp_input.txt"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        f.write(text)

    cmd = [
        "text2image",
        f"--font={font}",
        f"--text={tmp_txt}",
        f"--outputbase={output_base}",
        f"--ptsize={font_size}",
        "--margin=10",
        "--strip_unrenderable_words",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(result.stderr)

    tif_path = output_base + ".tif"
    png_path = output_base + ".png"

    if os.path.exists(tif_path):
        img = Image.open(tif_path).convert("RGB")
        img.save(png_path)
        os.remove(tif_path)
        return png_path

    return None

print("Generating clean images...")
records   = []
img_index = 0

for font, batch_texts in font_batches:
    print(f"  Font: {font} → {len(batch_texts)} images")

    for text in batch_texts:
        try:
            output_base = os.path.join(OUTPUT_DIR, str(img_index))
            png_path    = generate_with_text2image(text, font, output_base)

            if png_path:
                crop_and_pad(png_path)
                records.append((png_path, text))
                img_index += 1

            if img_index % 500 == 0:
                print(f"    {img_index} images generated...")

        except Exception as e:
            print(f"    Skipped [{font}]: {e}")
            continue

# Cleanup temp file
if os.path.exists("tmp_input.txt"):
    os.remove("tmp_input.txt")

# ── STEP 7: Save labels CSV ───────────────────────────────
with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "text"])
    for img_path, label in records:
        writer.writerow([img_path, label])

print(f"\n✅ Done! {img_index} clean images saved to '{OUTPUT_DIR}'")
print(f"✅ Labels saved to '{LABELS_FILE}'")
