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


def blur_interpolate(tensor, target_size, kernel_size=5, sigma=1.0):
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
    import os
import shutil

def copy_traj_data(input_root: str, output_root: str):
    """
    Recursively search for 'traj_data.pkl' or 'traj_data.json' under input_root
    and copy each found file into the corresponding location under output_root,
    preserving the relative folder structure.
    """
    for dirpath, dirnames, filenames in os.walk(input_root):
        for name in filenames:
            if name in ("traj_data.pkl", "traj_data.json"):
                # Compute relative path from input_root
                rel_dir = os.path.relpath(dirpath, input_root)
                dest_dir = os.path.join(output_root, rel_dir)
                os.makedirs(dest_dir, exist_ok=True)

                src = os.path.join(dirpath, name)
                dst = os.path.join(dest_dir, name)

                # Skip if already exists and is same size
                if os.path.exists(dst) and os.path.getsize(src) == os.path.getsize(dst):
                    print(f"Skipping (already copied): {src}")
                    continue

                shutil.copy2(src, dst)
                print(f"Copied {src} → {dst}")

class ImageBatchProcessor:
    def __init__(self, input_dir, output_dir, target_size=(320, 240),
                 downsample_factor=2, upsample_factor=2, sample_method='up',
                 interpolate=True, crop_s=16, large=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.downsample_factor = downsample_factor
        self.upsample_factor = upsample_factor
        self.sample_method = sample_method
        self.interpolate = interpolate
        self.crop_s = crop_s
        self.large   = large

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
        Recursively iterates through all subfolders (within self.input_dir)
        that contain image files (png, jpg, jpeg, bmp), calling
        load_folder, process_images, and save_folder for each.
        Processed outputs are saved to the corresponding subfolder under self.output_dir,
        preserving the relative folder structure.
        """
        # Use os.walk to recursively find directories that contain at least one image.
        folder_list = []
        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            image_files = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if image_files:
                folder_list.append(dirpath)
        
        folder_list.sort()
        print(f"Found {len(folder_list)} subfolders with images in {self.input_dir}.")

        for i, folder in enumerate(folder_list):
            # Compute the relative folder path with respect to self.input_dir
            rel_folder = os.path.relpath(folder, self.input_dir)
            in_subfolder_path = folder
            out_subfolder_path = os.path.join(self.output_dir, rel_folder)
            print(f"\n=== Processing subfolder {i+1}/{len(folder_list)}: {rel_folder} ===")

             # ——— large-mode fast path ———
            if self.large:
                os.makedirs(out_subfolder_path, exist_ok=True)
                target_w, target_h = self.target_size

                for fn in sorted(os.listdir(in_subfolder_path)):
                    if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue

                    src = os.path.join(in_subfolder_path, fn)
                    dst = os.path.join(out_subfolder_path, fn)
                    with Image.open(src) as img:
                        W, H = img.size
                        desired_ar = 4/3

                        # center-crop to 4:3
                        if W / H > desired_ar:
                            newW = int(H * desired_ar)
                            left = (W - newW) // 2
                            img = img.crop((left, 0, left + newW, H))
                        else:
                            newH = int(W / desired_ar)
                            top  = (H - newH) // 2
                            img = img.crop((0, top, W, top + newH))

                        # Lanczos downsample
                        img.resize((target_w, target_h), Image.LANCZOS).save(dst)

                # copy any traj_data.json from this folder or its parent
                for cand in (
                    os.path.join(in_subfolder_path,    "traj_data.json"),
                    os.path.join(os.path.dirname(folder), "traj_data.json")
                ):
                    if os.path.isfile(cand):
                        shutil.copy2(cand, out_subfolder_path)
                        break

                print(f"[large] done {rel_folder}: {len(os.listdir(out_subfolder_path))} images")
                continue
            # ——————————————————————————————
        
            # 1) Load images from this subfolder
            self.load_folder(in_subfolder_path)
            print(f"Loaded {len(self.images)} images from {in_subfolder_path}")
            
            '''
            # 1.1) Filter out images already processed
            initial_count = len(self.images)
            self.images = {rel_path: img for rel_path, img in self.images.items()
                        if not os.path.exists(os.path.join(out_subfolder_path, rel_path))}
            skipped_count = initial_count - len(self.images)
            if skipped_count > 0:
                print(f"Skipping {skipped_count} images already processed in {out_subfolder_path}.")

            if not self.images:
                print("All images in this subfolder have already been processed. Skipping processing.")
                continue
            #'''

            # 2) Process images using your existing pipeline
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

    def save_folder(self, input_subfolder_path):
        """
        Saves self.processed_images into a new folder in the same parent as `input_subfolder_path`.
        For example, if input_subfolder_path is:
        /path/to/traj_folder/_camera_rgb_image_raw_compressed
        then we create:
        /path/to/traj_folder/rgb_320x240_camera_rgb_image_raw_compressed
        and place images there.
        """

        # 1) Identify the parent trajectory folder and subfolder name
        parent_dir = os.path.dirname(input_subfolder_path)
        base_name = os.path.basename(input_subfolder_path)  # e.g. '_camera_rgb_image_raw_compressed'

        # 2) Extract a short camera label (adjust logic as needed)
        #    For instance, remove the leading underscore and take the second token:
        #    "_camera_rgb_image_raw_compressed" -> "camera_rgb_image_raw_compressed"
        #    split -> ["camera","rgb","image","raw","compressed"]
        #    camera_name -> "rgb"
        name_no_underscore = base_name.lstrip('_')
        parts = name_no_underscore.split('_')
        if parts[0] == "spot" and len(parts) >= 3:
            camera_name = parts[2]
        elif parts[0] == "image" and len(parts) >= 3:
            camera_name = "rgb"
        elif parts[0] == "left" and len(parts) >= 3:
            camera_name = "rgb"
        elif len(parts) >= 2:
            camera_name = parts[1]
        else:
            camera_name = "unknown"

        # 3) Build the new folder name
        new_folder_name = os.path.basename(input_subfolder_path)
        final_output_folder = os.path.join(parent_dir, new_folder_name)

        print(f"Saving processed images into {final_output_folder}")
        os.makedirs(final_output_folder, exist_ok=True)

        # 4) Save each processed image
        #    Optionally rename the files with a prefix
        for rel_path, preds in self.processed_images.items():
            # Original base name
            original_name = os.path.basename(rel_path)
            # Optionally prefix the file name: "rgb_320x240_<original>"
            # Or just keep the original_name
            new_filename = f"{camera_name}_{self.target_size[0]}x{self.target_size[1]}_{original_name}"
            out_path = os.path.join(final_output_folder, new_filename)

            # Actually save the processed image
            ImageLoader.save_image(preds, out_path)
            print(f"Saved processed image to {out_path}")
        
        # 5) Copy camera_info.json if present in the input subfolder to the new folder.
        json_src = os.path.join(input_subfolder_path, "camera_info.json")
        if os.path.isfile(json_src):
            json_dst = os.path.join(final_output_folder, "camera_info.json")
            shutil.copy2(json_src, json_dst)
            print(f"Copied {json_src} to {json_dst}")

        '''
        # 6) Delete the original input subfolder (raw trajectory folder).
        #     CAREFUL: This permanently deletes the raw images!
        try:
            shutil.rmtree(input_subfolder_path)
            print(f"Deleted original raw folder: {input_subfolder_path}")
        except Exception as e:
            print(f"Error deleting {input_subfolder_path}: {e}")
        #'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process images using a custom pipeline with optional Gaussian blur and interpolation."
    )
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for processed images")
    parser.add_argument("--width", type=int, default=320, help="Target width after interpolation")
    parser.add_argument("--height", type=int, default=240, help="Target height after interpolation")
    parser.add_argument("--method", type=str, default="up", choices=['up', 'down'], help="Sampling method")
    parser.add_argument("--crop", type=int, default=0, help="Pixels to crop from top and bottom")
    parser.add_argument("--downsample-factor", type=int, default=2, help="Factor for downsampling")
    parser.add_argument("--upsample-factor", type=int, default=2, help="Factor for upsampling")
    parser.add_argument("--large", action="store_true", help="direct 4:3 crop & Lanczos downsample for 3 big input sizes")
    args = parser.parse_args()

    processor = ImageBatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=(args.width, args.height),
        downsample_factor=args.downsample_factor,
        upsample_factor=args.upsample_factor,
        sample_method=args.method,
        crop_s=args.crop,
        large=args.large
    )
    processor.process_subfolders()
    copy_traj_data(args.input_dir, args.output_dir)