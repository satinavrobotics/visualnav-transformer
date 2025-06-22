#!/usr/bin/env python3
"""
dataset_preprocess.py

A script to load all images from an input folder, process them using a pipeline that:
  1. Optionally crops the image.
  2. Applies a Gaussian blur and then resizes the image to a target size.
  3. Upsamples (or downscales) the image using either a pretrained super‑image model
     (which you could fine‑tune on your dataset) or a fallback PIL-based resize.

Processed images are saved to the output folder preserving the folder structure.
"""

import os
import argparse
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import shutil

# 1. Enhanced CUDA Diagnostics
print("\n=== CUDA Diagnostic Report ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test basic CUDA functionality
        test_tensor = torch.empty(3, device='cuda')
        print("Basic CUDA tensor creation successful")
        del test_tensor
        
        # Test memory allocation
        big_tensor = torch.empty(1024, 1024, 1024, device='cuda')  # ~4GB
        print("Large memory allocation successful")
        del big_tensor
        
    except Exception as e:
        print(f"CUDA FAILURE: {str(e)}")
        torch.cuda.is_available = lambda: False  # Force disable CUDA

# 2. Safe Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nFinal computation device: {device}")

# 3. Super-Image Initialization with Enhanced Safety
USE_SUPER = False
super_image_error = None

if device.type == 'cuda':
    try:
        from super_image import EdsrModel, ImageLoader
        
        # Memory-safe model loading
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
            model = model.to(device)
            
            # Verify model functionality
            test_input = torch.randn(1, 3, 64, 64).to(device)
            with torch.no_grad():
                output = model(test_input)
            print(f"Model test output shape: {output.shape}")
            
            USE_SUPER = True
            print("Super-image initialized successfully")
            
    except Exception as e:
        super_image_error = str(e)
        USE_SUPER = False
        print(f"Super-image initialization failed: {super_image_error}")


def crop(image, crop_s=16):
    """Crop crop_s pixels from the top and bottom if the image height > 270."""
    w, h = image.size
    if h > 270:
        return image.crop((0, crop_s, w, h - crop_s))
    return image


def blur_interpolate(tensor, target_size, kernel_size=5, sigma=0.1):
    """
    Apply Gaussian blur and resize using torch.
    Returns PIL Image for compatibility with super_image.
    """
    # Apply Gaussian blur
    blur = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    blurred = blur(tensor)
    
    '''
    # Interpolate to target size
    resized = F.interpolate(
        blurred, 
        size=(target_size[1], target_size[0]), 
        mode='bilinear', 
        align_corners=False
    )
    '''
    
    # Convert back to PIL Image
    return blurred


def torch_upsample(tensor, factor=2):
    # tensor shape: [1, C, H, W]
    h, w = tensor.shape[2], tensor.shape[3]
    upsampled = F.interpolate(tensor, size=(h * factor, w * factor), mode='bilinear', align_corners=False)
    return upsampled  # remains a tensor


def torch_downsample(tensor, factor=2):
    # tensor shape: [1, C, H, W]
    h, w = tensor.shape[2], tensor.shape[3]
    downsampled = F.interpolate(tensor, size=(h // factor, w // factor), mode='bilinear', align_corners=False)
    return downsampled  # remains a tensor


def copy_traj_data(input_folder, output_folder):
    """
    Recursively search for 'traj_data.pkl' in input_folder and copy each found file to the corresponding
    location in output_folder, preserving the relative folder structure.
    If a file already exists at the destination with the same size, it will be skipped.
    """
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file == "traj_data.pkl":
                rel_dir = os.path.relpath(root, input_folder)
                dest_dir = os.path.join(output_folder, rel_dir)
                os.makedirs(dest_dir, exist_ok=True)
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                # Check if the destination file already exists and has the same size
                if os.path.exists(dest_path) and os.path.getsize(src_path) == os.path.getsize(dest_path):
                    print(f"Skipping {src_path} as it is already copied.")
                    continue
                shutil.copy2(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
                

class ImageBatchProcessor:
    def __init__(self, input_dir, output_dir, target_size=(320, 240),
                 downsample_factor=2, upsample_factor=2, sample_method='up',
                 interpolate=True, crop_s=16):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.downsample_factor = downsample_factor
        self.upsample_factor = upsample_factor
        self.sample_method = sample_method
        self.interpolate = interpolate
        self.crop_s = crop_s

        # We re-create these dictionaries per subfolder
        self.images = {}
        self.processed_images = {}

    def load_folder(self, folder_path):
        """
        Loads images (png, jpg, jpeg, bmp) from a specific folder_path
        into self.images, clearing any previous images.
        """
        self.images.clear()  # start fresh
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                    full_path = os.path.join(root, file)
                    try:
                        img = Image.open(full_path).convert("RGB")
                        self.images[rel_path] = img
                    except Exception as e:
                        print(f"Warning: Could not load image {full_path}: {e}")

    def process_subfolders(self):
        """
        Iterates each immediate subfolder in self.input_dir, calling
        load_folder, process_images, and save_folder for each. Outputs
        go to a matching subfolder under self.output_dir.
        """
        subfolders = []
        for entry in os.scandir(self.input_dir):
            if entry.is_dir():
                subfolders.append(entry.name)

        subfolders.sort()
        print(f"Found {len(subfolders)} subfolders in {self.input_dir}.")

        for i, sub in enumerate(subfolders):
            in_subfolder_path = os.path.join(self.input_dir, sub)
            out_subfolder_path = os.path.join(self.output_dir, sub)
            print(f"\n=== Processing subfolder {i+1}/{len(subfolders)}: {sub} ===")

            # 1) Load images from this subfolder
            self.load_folder(in_subfolder_path)
            print(f"Loaded {len(self.images)} images from {in_subfolder_path}")
            
            # 1.1) Filter out images that have already been processed
            initial_count = len(self.images)
            self.images = {rel_path: img for rel_path, img in self.images.items()
                        if not os.path.exists(os.path.join(out_subfolder_path, rel_path))}
            skipped_count = initial_count - len(self.images)
            if skipped_count > 0:
                print(f"Skipping {skipped_count} images already processed in {out_subfolder_path}.")

            if not self.images:
                print("All images in this subfolder have already been processed. Skipping processing.")
                continue

            # 2) Process them (your existing pipeline)
            self.process_images()

            # 3) Save results into matching output subfolder
            self.save_folder(out_subfolder_path)
            print(f"Saved processed images to {out_subfolder_path}")

            # 4) Optionally clear GPU memory each time
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nAll subfolders processed successfully!")

    def process_all(self):
        """
        If you still want a single 'process_all()' for the entire directory at once,
        you can keep your old approach here.
        """
        self.load_folder(self.input_dir)
        print(f"Loaded {len(self.images)} images from {self.input_dir}")
        self.process_images()
        self.save_folder(self.output_dir)
        print(f"Processed images have been saved to {self.output_dir}")

    def process_images(self):
        """Process each image through the pipeline"""
        self.processed_images.clear()
        for rel_path, img in self.images.items():
            try:
                # Step 1: Crop (operates on PIL Image)
                cropped = crop(img, self.crop_s)
                
                with torch.no_grad():
                    if self.sample_method == 'up':
                        if USE_SUPER:
                            # Load image and run through super_image model
                            inputs = ImageLoader.load_image(cropped).to(device)
                            preds = model(inputs)
                            if self.interpolate:
                                preds_intp = blur_interpolate(preds, self.target_size)
                            else:
                                preds_intp = inputs
                            # Move prediction to CPU
                            processed = preds_intp.cpu()
                            # Delete intermediate tensors
                            del inputs, preds, preds_intp
                        else:
                            cropped_tensor = ImageLoader.load_image(cropped).to(device)
                            processed = torch_upsample(cropped_tensor, self.upsample_factor)
                            if self.interpolate:
                                processed_intp = blur_interpolate(processed, self.target_size)
                            else:
                                processed_intp = cropped_tensor
                            processed = processed_intp.cpu()
                            del cropped_tensor, processed_intp
                    else:
                        cropped_tensor = ImageLoader.load_image(cropped).to(device)
                        # print(cropped_tensor.shape)
                        processed = torch_downsample(cropped_tensor, self.upsample_factor)
                        # print(processed.shape)
                        if self.interpolate:
                            processed_intp = blur_interpolate(processed, self.target_size)
                            # print(processed_intp.shape)
                        else:
                            processed_intp = cropped_tensor
                        processed = processed_intp.cpu()
                        del cropped_tensor, processed_intp
                        
                        # Clear GPU cache and delete temporary variables
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        pass

                self.processed_images[rel_path] = processed
    
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")     

    def save_folder(self, folder_path):
        """Saves self.processed_images to the specified folder_path, preserving relative structure."""
        for rel_path, preds in self.processed_images.items():
            out_path = os.path.join(folder_path, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            ImageLoader.save_image(preds, out_path)
            print(f"Saved processed image to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process images using a custom pipeline with optional Gaussian blur and interpolation."
    )
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for processed images")
    parser.add_argument("--width", type=int, default=320, help="Target width after interpolation")
    parser.add_argument("--height", type=int, default=240, help="Target height after interpolation")
    parser.add_argument("--method", type=str, default="up", choices=['up', 'down'], help="Sampling method")
    parser.add_argument("--crop", type=int, default=16, help="Pixels to crop from top and bottom")
    parser.add_argument("--downsample-factor", type=int, default=2, help="Factor for downsampling")
    parser.add_argument("--upsample-factor", type=int, default=2, help="Factor for upsampling")
    args = parser.parse_args()

    processor = ImageBatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=(args.width, args.height),
        downsample_factor=args.downsample_factor,
        upsample_factor=args.upsample_factor,
        sample_method=args.method,
        crop_s=args.crop
    )
    processor.process_subfolders()
    # copy_traj_data(args.input_dir, args.output_dir)