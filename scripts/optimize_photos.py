from __future__ import annotations

import pathlib
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "assets" / "images"
OUTPUT_DIR = ROOT / "assets" / "images" / "optimized"

SIZES = [480, 960, 1600]
WEBP_QUALITY = 80


def iter_source_images():
    for path in SOURCE_DIR.rglob("*"):
        if not path.is_file():
            continue
        if OUTPUT_DIR in path.parents:
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        yield path


def build_output_path(src_path: pathlib.Path, width: int) -> pathlib.Path:
    rel = src_path.relative_to(SOURCE_DIR)
    stem = rel.with_suffix("")
    return OUTPUT_DIR / f"{stem}-{width}w.webp"


def save_resized_webp(src_path: pathlib.Path):
    with Image.open(src_path) as img:
        original_width, original_height = img.size
        for width in SIZES:
            scale = width / float(original_width)
            height = int(original_height * scale)
            resized = img.resize((width, height), Image.LANCZOS)
            out_path = build_output_path(src_path, width)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            resized.save(out_path, "WEBP", quality=WEBP_QUALITY, method=6)


def main():
    for image_path in iter_source_images():
        save_resized_webp(image_path)


if __name__ == "__main__":
    main()
