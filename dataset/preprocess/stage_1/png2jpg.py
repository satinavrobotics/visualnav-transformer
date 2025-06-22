#!/usr/bin/env python3
import os
import argparse
from PIL import Image

def png_to_jpg_high_fidelity(root_dir, remove_png=False, quality=100):
    """
    Recursively convert all .png files under `root_dir` to .jpg with:
      • quality=100
      • optimize=True
      • subsampling=0 (no chroma subsampling)
    """
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith('.png'):
                continue

            png_path = os.path.join(dirpath, fn)
            jpg_name = os.path.splitext(fn)[0] + '.jpg'
            jpg_path = os.path.join(dirpath, jpg_name)

            if os.path.exists(jpg_path):
                print(f"[SKIP] already exists: {jpg_path}")
                if remove_png:
                    os.remove(png_path)
                    print(f"[DEL] {png_path}")
                continue
            try:
                with Image.open(png_path) as im:
                    # RGBA → white background
                    if im.mode in ('RGBA', 'LA'):
                        bg = Image.new('RGB', im.size, (255,255,255))
                        bg.paste(im, mask=im.split()[-1])
                        rgb = bg
                    else:
                        rgb = im.convert('RGB')

                    rgb.save(
                        jpg_path,
                        format='JPEG',
                        quality=quality,
                        optimize=True,
                        subsampling=0
                    )

                print(f"[OK]  {png_path} → {jpg_path}")

                if remove_png:
                    os.remove(png_path)
                    print(f"[DEL] {png_path}")

            except Exception as e:
                print(f"[ERR] {png_path}: {e}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="High-fidelity PNG→JPEG converter"
    )
    p.add_argument("root_dir", help="Folder to scan for .png files")
    p.add_argument(
        "-r", "--remove", action="store_true",
        help="delete original .png after converting"
    )
    p.add_argument(
        "-q", "--quality", type=int, default=100,
        help="JPEG quality (1–100)"
    )
    args = p.parse_args()
    png_to_jpg_high_fidelity(
        args.root_dir,
        remove_png=args.remove,
        quality=args.quality
    )