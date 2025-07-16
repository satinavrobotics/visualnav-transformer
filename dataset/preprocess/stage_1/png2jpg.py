#!/usr/bin/env python3
import os
import argparse
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable optimize due to memory cost on large batches
OPTIMIZE = False

def convert_file(png_path, remove_png=False, quality=100):
    try:
        dirpath, fn = os.path.split(png_path)
        jpg_name = os.path.splitext(fn)[0] + '.jpg'
        jpg_path = os.path.join(dirpath, jpg_name)

        if os.path.exists(jpg_path):
            if remove_png:
                try:
                    os.remove(png_path)
                except Exception as e:
                    return f"[ERR] deleting {png_path}: {e}"
            return None

        with Image.open(png_path) as im:
            # Add white background if image has alpha channel
            if im.mode in ('RGBA', 'LA'):
                bg = Image.new('RGB', im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                rgb = bg
            else:
                rgb = im.convert('RGB')

            rgb.save(
                jpg_path,
                format='JPEG',
                quality=quality,
                optimize=OPTIMIZE,
                subsampling=0
            )

        if remove_png:
            try:
                os.remove(png_path)
            except Exception as e:
                return f"[ERR] deleting {png_path}: {e}"

        return None

    except Exception as e:
        return f"[ERR] converting {png_path}: {e}"

def png_to_jpg_high_fidelity(root_dir, remove_png=False, quality=100, num_workers=4):
    # Find all .png files recursively
    png_files = []
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith('.png'):
                png_files.append(os.path.join(dirpath, fn))

    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_file, path, remove_png, quality) for path in png_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Converting", unit="file"):
            err = f.result()
            if err:
                errors.append(err)

    # Only print errors
    if errors:
        print("\nErrors:")
        for err in errors:
            print(err)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="High-fidelity PNG→JPEG converter with concurrency.")
    parser.add_argument("root_dir", help="Folder to scan for .png files")
    parser.add_argument("-r", "--remove", action="store_true", help="Delete original .png after converting")
    parser.add_argument("-q", "--quality", type=int, default=100, help="JPEG quality (1–100)")
    parser.add_argument("-w", "--workers", type=int, default=20, help="Number of threads to use")
    args = parser.parse_args()

    png_to_jpg_high_fidelity(
        root_dir=args.root_dir,
        remove_png=args.remove,
        quality=args.quality,
        num_workers=args.workers
    )
