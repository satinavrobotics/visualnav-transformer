#!/usr/bin/env python3
import sys
from PIL import Image

def main(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Dimensions: {width} x {height}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python img_test.py <path_to_image>")
    else:
        main(sys.argv[1])