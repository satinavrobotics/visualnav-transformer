#!/usr/bin/env python3
"""
dataset_preprocess.py

Simplified dataset preprocessing script with automated parameter detection.
Processes images through a configurable pipeline with smart defaults.
"""

import os
import argparse
import yaml
import json
import math
import time
import gc
from pathlib import Path
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import shutil
import gc
import psutil
from typing import Dict, Tuple, Optional, Any, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from datetime import datetime


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'target_size': (320, 240),
        'blur_kernel': 'auto',
        'blur_sigma': 'auto',
        'apply_blur': True,
        'batch_size': 8,
        'num_workers': 4,
        'overwrite': False,
        'crop_from_top': False
    }


class AutoConfig:
    """Automatic configuration detection and management"""

    def __init__(self):
        self.device = self._detect_device()
        self.use_super_image = False  # Will be initialized lazily when needed
        self.super_image_initialized = False

    def _detect_device(self) -> torch.device:
        """Automatically detect and configure the best available device"""
        if not torch.cuda.is_available():
            print("Using CPU (CUDA not available)")
            return torch.device('cpu')

        try:
            # Test CUDA functionality
            test_tensor = torch.empty(100, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device('cuda')
        except Exception as e:
            print(f"CUDA test failed, falling back to CPU: {e}")
            return torch.device('cpu')

    def _init_super_image_if_needed(self, method: str) -> bool:
        """Initialize super-image model only when upsampling is needed"""
        if method != 'up' or self.super_image_initialized:
            return self.use_super_image

        if self.device.type != 'cuda':
            print("Super-image requires CUDA, using fallback processing")
            self.super_image_initialized = True
            return False

        try:
            from super_image import EdsrModel, ImageLoader
            global model, IL

            print("Initializing super-image model for upsampling...")
            torch.cuda.empty_cache()
            model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
            model = model.to(self.device)
            IL = ImageLoader

            # Quick test
            test_input = torch.randn(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                _ = model(test_input)
            del test_input
            torch.cuda.empty_cache()

            print("Super-image model initialized successfully")
            self.use_super_image = True
            self.super_image_initialized = True
            return True
        except Exception as e:
            print(f"Super-image initialization failed, using fallback: {e}")
            self.use_super_image = False
            self.super_image_initialized = True
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'target_size': (320, 240),
            'blur_kernel': 'auto',
            'blur_sigma': 'auto',
            'apply_blur': True,
            'batch_size': 8,
            'num_workers': 4
        }

    def analyze_first_image(self, input_dir: str, target_size: Tuple[int, int] = (320, 240)) -> Dict[str, Any]:
        """Analyze first image to infer optimal processing parameters"""
        # Find first image
        first_image_path = None
        for root, _, files in os.walk(input_dir):
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    first_image_path = os.path.join(root, file)
                    break
            if first_image_path:
                break

        if not first_image_path:
            print("No images found, using default parameters")
            return self.get_default_config()

        try:
            with Image.open(first_image_path) as img:
                original_size = img.size  # (width, height)
                print(f"Analyzing first image: {os.path.basename(first_image_path)} ({original_size[0]}x{original_size[1]})")

                # Calculate optimal parameters
                params = self._infer_processing_params(original_size, target_size)

                # Select base preset based on input size
                if original_size[0] > 1000 or original_size[1] > 800:
                    base_preset = 'high_res'
                elif original_size[0] < 400 or original_size[1] < 300:
                    base_preset = 'fast'
                else:
                    base_preset = 'standard'

                # Start with base config and apply inferred parameters
                config = self.get_default_config()
                config.update(params)

                print(f"Inferred parameters: crop={params['crop_pixels']}px, "
                      f"factor={params['upsample_factor']}, method={params['method']}")
                print(f"Base preset: {base_preset}, target: {target_size[0]}x{target_size[1]}")

                return config

        except Exception as e:
            print(f"Error analyzing first image: {e}, using default parameters")
            return self.get_default_config()

    def _infer_processing_params(self, original_size: Tuple[int, int], target_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Infer optimal processing parameters using precise multiplication approach:
        1. Calculate exact multiplication factor to reach or exceed target size
        2. Upsample at that rate
        3. Determine crop based on difference between upsampled and target size
        """
        orig_w, orig_h = original_size
        target_w, target_h = target_size

        # Calculate multiplication factors needed to reach or exceed target size
        factor_w = target_w / orig_w  # Factor needed for width
        factor_h = target_h / orig_h  # Factor needed for height

        # Use the larger factor to ensure both dimensions reach or exceed target
        required_factor = max(factor_w, factor_h)

        # Determine method and calculate appropriate factor
        if required_factor >= 1.0:
            # Need to upsample - round up to nearest integer
            method = 'up'
            upsample_factor = max(1, int(np.ceil(required_factor)))
        else:
            # Need to downsample - calculate reduction factor
            method = 'down'
            # For downsampling, we want the factor that when we divide by it,
            # we get close to the target size
            reduction_factor = 1.0 / required_factor
            upsample_factor = max(1, int(np.ceil(reduction_factor)))

        # Calculate final dimensions after scaling
        if method == 'up':
            upsampled_w = orig_w * upsample_factor
            upsampled_h = orig_h * upsample_factor
        else:  # method == 'down'
            upsampled_w = orig_w // upsample_factor
            upsampled_h = orig_h // upsample_factor

        # Calculate crop needed to reach exact target size
        # Crop is the excess that needs to be removed from each side
        crop_w = max(0, (upsampled_w - target_w) // 2)
        crop_h = max(0, (upsampled_h - target_h) // 2)

        # Use the larger crop dimension as the crop_pixels parameter
        crop_pixels = max(crop_w, crop_h)

        # Log the calculation for debugging
        print(f"Size calculation: {orig_w}x{orig_h} → {target_w}x{target_h}")
        print(f"Required factor: {required_factor:.2f}")
        print(f"Upsample factor: {upsample_factor}, Method: {method}")
        print(f"Upsampled size: {upsampled_w}x{upsampled_h}")
        print(f"Crop needed: {crop_pixels}px (w:{crop_w}, h:{crop_h})")

        return {
            'crop_pixels': crop_pixels,
            'upsample_factor': upsample_factor,
            'method': method,
            'target_size': target_size,
            'inferred_from_image': True,
            # Additional debug info
            'original_size': original_size,
            'upsampled_size': (upsampled_w, upsampled_h),
            'required_factor': required_factor
        }

    def auto_detect_dataset_params(self, input_dir: str, target_size: Tuple[int, int] = None) -> Dict[str, Any]:
        """Automatically detect optimal parameters based on first image analysis"""
        if target_size is None:
            target_size = (320, 240)  # Default target size

        return self.analyze_first_image(input_dir, target_size)

# Initialize global configuration
auto_config = AutoConfig()


class ProgressLogger:
    """Handles progress logging and resumption for dataset preprocessing"""

    def __init__(self, output_dir: Path, input_dir: Path):
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.progress_file = self.output_dir / '.preprocessing_progress.pkl'
        self.log_file = self.output_dir / 'preprocessing.log'

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load progress state
        self.progress_state = self._load_progress()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup text logging to file"""
        # Open log file in append mode
        self.log_handle = open(self.log_file, 'a', encoding='utf-8')
        self._log(f"\n{'='*60}")
        self._log(f"Preprocessing session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Input directory: {self.input_dir}")
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"{'='*60}")

    def _log(self, message: str):
        """Write message to log file and print to console"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_handle.write(log_message + '\n')
        self.log_handle.flush()

    def _load_progress(self) -> Dict[str, Any]:
        """Load existing progress state or create new one"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    state = pickle.load(f)
                print(f"📋 Loaded existing progress from {self.progress_file}")
                return state
            except Exception as e:
                print(f"⚠️  Could not load progress file: {e}. Starting fresh.")

        # Create new progress state
        return {
            'completed_folders': set(),
            'current_folder': None,
            'current_folder_progress': 0,
            'total_folders': 0,
            'total_images_processed': 0,
            'start_time': time.time(),
            'last_update': time.time()
        }

    def _save_progress(self):
        """Save current progress state to file"""
        self.progress_state['last_update'] = time.time()
        try:
            # Write to temporary file first, then rename for atomic operation
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(self.progress_state, f)
            temp_file.rename(self.progress_file)
        except Exception as e:
            print(f"⚠️  Could not save progress: {e}")

    def should_skip_folder(self, folder_path: Path) -> bool:
        """Check if folder should be skipped (already completed)"""
        rel_path = str(folder_path.relative_to(self.input_dir))
        return rel_path in self.progress_state['completed_folders']

    def start_folder(self, folder_path: Path, total_images: int):
        """Mark start of processing a folder"""
        rel_path = str(folder_path.relative_to(self.input_dir))
        self.progress_state['current_folder'] = rel_path
        self.progress_state['current_folder_progress'] = 0
        self._log(f"📁 Starting folder: {rel_path} ({total_images} images)")
        self._save_progress()

    def update_folder_progress(self, images_processed: int):
        """Update progress within current folder"""
        self.progress_state['current_folder_progress'] = images_processed
        self.progress_state['total_images_processed'] += images_processed
        self._save_progress()

    def complete_folder(self, folder_path: Path, images_processed: int):
        """Mark folder as completed"""
        rel_path = str(folder_path.relative_to(self.input_dir))
        self.progress_state['completed_folders'].add(rel_path)
        self.progress_state['current_folder'] = None
        self.progress_state['current_folder_progress'] = 0

        elapsed = time.time() - self.progress_state['start_time']
        completed_count = len(self.progress_state['completed_folders'])
        total_count = self.progress_state['total_folders']

        self._log(f"✅ Completed folder: {rel_path} ({images_processed} images)")
        self._log(f"📊 Progress: {completed_count}/{total_count} folders "
                 f"({completed_count/total_count*100:.1f}%) - "
                 f"Elapsed: {elapsed/60:.1f}min")

        self._save_progress()

    def set_total_folders(self, total: int):
        """Set total number of folders to process"""
        self.progress_state['total_folders'] = total
        self._save_progress()

    def get_resume_info(self) -> Dict[str, Any]:
        """Get information about what can be resumed"""
        completed = len(self.progress_state['completed_folders'])
        total = self.progress_state['total_folders']

        if completed == 0:
            return {'can_resume': False, 'message': 'Starting fresh processing'}

        if completed == total:
            return {'can_resume': False, 'message': 'All folders already completed'}

        current = self.progress_state.get('current_folder')
        if current:
            progress = self.progress_state.get('current_folder_progress', 0)
            return {
                'can_resume': True,
                'message': f'Resuming from folder {completed+1}/{total}. '
                          f'Last folder "{current}" had {progress} images processed.',
                'completed_folders': completed,
                'total_folders': total,
                'current_folder': current
            }
        else:
            return {
                'can_resume': True,
                'message': f'Resuming processing. {completed}/{total} folders completed.',
                'completed_folders': completed,
                'total_folders': total
            }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'log_handle') and not self.log_handle.closed:
            self._log(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_handle.close()

    def __del__(self):
        """Ensure cleanup on destruction"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup


class ImageDataset(Dataset):
    """Custom Dataset for loading images (processing happens in batch)"""

    def __init__(self, image_files: List[Path], processor: 'ImageProcessor'):
        """
        Args:
            image_files: List of image file paths
            processor: ImageProcessor instance for processing images
        """
        self.image_files = image_files
        self.processor = processor
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            Tuple of (raw_tensor, original_file_path_str) - processing happens later in batch
        """
        img_file = self.image_files[idx]

        try:
            with Image.open(img_file) as img:
                img = img.convert("RGB")
                # Only convert to tensor, don't process yet (for true batch processing)
                transform = torchvision.transforms.ToTensor()
                tensor = transform(img)
                # Explicitly delete the PIL image to free memory immediately
                del img
                return tensor, str(img_file)  # Convert Path to string for DataLoader compatibility
        except Exception as e:
            # Return a zero tensor and the file path for error handling
            target_size = self.processor.config.get('target_size', (320, 240))
            zero_tensor = torch.zeros(3, target_size[1], target_size[0])
            return zero_tensor, str(img_file)  # Convert Path to string for DataLoader compatibility



class ImageProcessor:
    """Simplified image processing pipeline with smart defaults"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = auto_config.device

        # Initialize super-image only if we're upsampling
        method = config.get('method', 'up')
        self.use_super_image = auto_config._init_super_image_if_needed(method)

        # Cache for blur transforms to avoid recreation
        self.blur_cache = {}

        # Initialize blur transform (will be updated dynamically for auto mode)
        self._init_blur_transform()

    def _init_blur_transform(self):
        """Initialize blur transform with default values (will be updated for auto mode)"""
        kernel = self.config.get('blur_kernel', 5)
        sigma = self.config.get('blur_sigma', 1.0)

        # Use defaults if auto mode (will be recalculated per operation)
        if kernel == 'auto':
            kernel = 5
        if sigma == 'auto':
            sigma = 1.0

        self.blur_transform = torchvision.transforms.GaussianBlur(kernel, sigma)

    def _calculate_auto_blur_params(self, scaling_factor: float, is_upsampling: bool) -> Tuple[int, float]:
        """Calculate optimal blur parameters based on scaling factor and direction

        Fine-tuned blur amounts for optimal quality:
        - Downsampling: sigma = 0.25 × s (was 0.3 × s)
        - Upsampling: sigma = 0.15 × s (was 0.2 × s)
        - Maximum sigma capped at 0.7 to prevent excessive blur
        """
        if is_upsampling:
            # For upsampling: sigma = 0.15 × s (reduced from 0.2)
            sigma = 0.15 * scaling_factor
        else:
            # For downsampling: sigma = 0.25 × s (reduced from 0.3)
            sigma = 0.25 * scaling_factor

        # Cap sigma to prevent excessive blur
        sigma = min(sigma, 0.7)  # Maximum sigma of 0.7 (reduced from 0.8)

        # Kernel size: 2 × ceil(3 × sigma) + 1 (must be odd)
        kernel_size = 2 * math.ceil(3 * sigma) + 1

        return kernel_size, sigma

    def _get_blur_transform(self, scaling_factor: float = 1.0, is_upsampling: bool = True) -> torchvision.transforms.GaussianBlur:
        """Get cached blur transform with auto-calculated parameters if needed"""
        kernel = self.config.get('blur_kernel', 5)
        sigma = self.config.get('blur_sigma', 1.0)

        # Calculate auto parameters if needed
        if kernel == 'auto' or sigma == 'auto':
            auto_kernel, auto_sigma = self._calculate_auto_blur_params(scaling_factor, is_upsampling)
            if kernel == 'auto':
                kernel = auto_kernel
            if sigma == 'auto':
                sigma = auto_sigma

        # Use cached transform if available
        cache_key = (kernel, sigma)
        if cache_key not in self.blur_cache:
            self.blur_cache[cache_key] = torchvision.transforms.GaussianBlur(kernel, sigma)

        return self.blur_cache[cache_key]

    def smart_crop(self, image: Image.Image) -> Image.Image:
        """Smart cropping based on image aspect ratio and size"""
        w, h = image.size
        crop_pixels = self.config.get('crop_pixels', 16)
        crop_from_top = self.config.get('crop_from_top', False)

        # Only crop if image is tall enough and has reasonable aspect ratio
        if h > 270 and crop_pixels > 0:
            if crop_from_top:
                # Crop only from top: remove crop_pixels*2 from top only
                return image.crop((0, crop_pixels * 2, w, h))
            else:
                # Default: crop evenly from top and bottom
                return image.crop((0, crop_pixels, w, h - crop_pixels))
        return image

    def process_tensor(self, tensor: torch.Tensor, target_size: Tuple[int, int], skip_blur: bool = False) -> torch.Tensor:
        """Process tensor through resize pipeline with proper anti-aliasing"""
        current_size = (tensor.shape[3], tensor.shape[2])  # W, H

        if current_size != target_size:
            # Determine if we're downsampling and calculate scaling factor
            current_pixels = current_size[0] * current_size[1]
            target_pixels = target_size[0] * target_size[1]
            is_downsampling = target_pixels < current_pixels

            # Calculate scaling factor for auto blur
            scaling_factor = math.sqrt(target_pixels / current_pixels) if current_pixels > 0 else 1.0
            if is_downsampling:
                scaling_factor = 1.0 / scaling_factor  # Invert for downsampling factor

            # Apply anti-aliasing blur BEFORE downsampling to prevent aliasing
            if is_downsampling and self.config.get('apply_blur', True) and not skip_blur:
                blur_transform = self._get_blur_transform(scaling_factor, is_upsampling=False)
                tensor = blur_transform(tensor)

            # Resize to target
            tensor = F.interpolate(
                tensor,
                size=(target_size[1], target_size[0]),  # H, W for interpolate
                mode='bilinear',
                align_corners=False
            )

            # Apply smoothing blur AFTER upsampling to reduce interpolation artifacts
            if not is_downsampling and self.config.get('apply_blur', True) and not skip_blur:
                blur_transform = self._get_blur_transform(scaling_factor, is_upsampling=True)
                tensor = blur_transform(tensor)

        return tensor

    def scale_tensor(self, tensor: torch.Tensor, method: str, factor: int) -> torch.Tensor:
        """Scale tensor up or down by factor with proper anti-aliasing"""
        h, w = tensor.shape[2], tensor.shape[3]

        if method == 'up':
            new_size = (h * factor, w * factor)
            # Upsample first
            scaled = F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)
            # Apply smoothing blur after upsampling if configured
            if self.config.get('apply_blur', True):
                blur_transform = self._get_blur_transform(float(factor), is_upsampling=True)
                scaled = blur_transform(scaled)
            return scaled
        else:  # down
            new_size = (h // factor, w // factor)
            # Apply anti-aliasing blur before downsampling if configured
            if self.config.get('apply_blur', True):
                blur_transform = self._get_blur_transform(float(factor), is_upsampling=False)
                tensor = blur_transform(tensor)
            # Then downsample
            return F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Main image processing pipeline for single image"""
        # Step 1: Smart crop
        cropped = self.smart_crop(image)

        # Step 2: Convert to tensor and move to device
        if self.use_super_image:
            tensor = IL.load_image(cropped).to(self.device)
        else:
            # Fallback tensor conversion
            transform = torchvision.transforms.ToTensor()
            tensor = transform(cropped).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Step 3: Apply super-resolution or scaling
            if self.use_super_image and self.config.get('method') == 'up':
                # Super-resolution model handles upsampling
                processed = model(tensor)
                # Final resize with blur (super-resolution may need smoothing)
                target_size = self.config.get('target_size', (320, 240))
                processed = self.process_tensor(processed, target_size, skip_blur=False)
            else:
                # Regular scaling with proper anti-aliasing
                processed = self.scale_tensor(
                    tensor,
                    self.config.get('method', 'up'),
                    self.config.get('upsample_factor', 2)
                )
                # Final resize without additional blur (already handled in scale_tensor)
                target_size = self.config.get('target_size', (320, 240))
                processed = self.process_tensor(processed, target_size, skip_blur=True)

            # Move to CPU for saving
            result = processed.cpu()

            # Clean up GPU memory
            del tensor, processed
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            return result

    def process_tensor_batch(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """Process a batch of tensors efficiently (true batch processing)"""
        # Move batch to device
        batch_tensors = batch_tensors.to(self.device)

        with torch.no_grad():
            # Step 1: Smart crop all images in batch
            batch_cropped = self._smart_crop_batch(batch_tensors)

            # Step 2: Apply super-resolution or scaling to entire batch
            if self.use_super_image and self.config.get('method') == 'up':
                # Super-resolution model handles upsampling
                processed = model(batch_cropped)
                # Final resize with blur (super-resolution may need smoothing)
                target_size = self.config.get('target_size', (320, 240))
                processed = self._process_tensor_batch_internal(processed, target_size, skip_blur=False)
            else:
                # Regular scaling with proper anti-aliasing
                processed = self._scale_tensor_batch(
                    batch_cropped,
                    self.config.get('method', 'up'),
                    self.config.get('upsample_factor', 2)
                )
                # Final resize without additional blur (already handled in scale_tensor)
                target_size = self.config.get('target_size', (320, 240))
                processed = self._process_tensor_batch_internal(processed, target_size, skip_blur=True)

            # Move to CPU for saving and immediately clean up GPU tensors
            result = processed.cpu()

            # Clean up GPU tensors
            del batch_tensors, batch_cropped, processed
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete

            return result

    def _smart_crop_batch(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """Apply smart crop to entire batch"""
        crop_pixels = self.config.get('crop_pixels', 0)
        crop_from_top = self.config.get('crop_from_top', False)

        if crop_pixels > 0:
            h = batch_tensors.shape[2]
            if crop_from_top:
                # Crop only from top: remove crop_pixels*2 from top only
                return batch_tensors[:, :, crop_pixels*2:, :]
            else:
                # Default: crop evenly from top and bottom
                return batch_tensors[:, :, crop_pixels:h-crop_pixels, :]
        return batch_tensors

    def _scale_tensor_batch(self, batch_tensors: torch.Tensor, method: str, factor: int) -> torch.Tensor:
        """Scale entire batch up or down by factor with proper anti-aliasing"""
        h, w = batch_tensors.shape[2], batch_tensors.shape[3]

        if method == 'up':
            new_size = (h * factor, w * factor)
            # Upsample first
            scaled = F.interpolate(batch_tensors, size=new_size, mode='bilinear', align_corners=False)
            # Apply smoothing blur after upsampling if configured
            if self.config.get('apply_blur', True):
                blur_transform = self._get_blur_transform(float(factor), is_upsampling=True)
                scaled = blur_transform(scaled)
            return scaled
        else:  # down
            new_size = (h // factor, w // factor)
            # Apply anti-aliasing blur before downsampling if configured
            if self.config.get('apply_blur', True):
                blur_transform = self._get_blur_transform(float(factor), is_upsampling=False)
                batch_tensors = blur_transform(batch_tensors)
            # Then downsample
            return F.interpolate(batch_tensors, size=new_size, mode='bilinear', align_corners=False)

    def _process_tensor_batch_internal(self, batch_tensors: torch.Tensor, target_size: Tuple[int, int], skip_blur: bool = False) -> torch.Tensor:
        """Process tensor batch through resize pipeline with proper anti-aliasing"""
        current_size = (batch_tensors.shape[3], batch_tensors.shape[2])  # W, H

        if current_size != target_size:
            # Determine if we're downsampling and calculate scaling factor
            current_pixels = current_size[0] * current_size[1]
            target_pixels = target_size[0] * target_size[1]
            is_downsampling = target_pixels < current_pixels

            # Calculate scaling factor for auto blur
            scaling_factor = math.sqrt(target_pixels / current_pixels) if current_pixels > 0 else 1.0
            if is_downsampling:
                scaling_factor = 1.0 / scaling_factor  # Invert for downsampling factor

            # Apply anti-aliasing blur BEFORE downsampling to prevent aliasing
            if is_downsampling and self.config.get('apply_blur', True) and not skip_blur:
                blur_transform = self._get_blur_transform(scaling_factor, is_upsampling=False)
                batch_tensors = blur_transform(batch_tensors)

            # Resize to target
            batch_tensors = F.interpolate(
                batch_tensors,
                size=(target_size[1], target_size[0]),  # H, W for interpolate
                mode='bilinear',
                align_corners=False
            )

            # Apply smoothing blur AFTER upsampling to reduce interpolation artifacts
            if not is_downsampling and self.config.get('apply_blur', True) and not skip_blur:
                blur_transform = self._get_blur_transform(scaling_factor, is_upsampling=True)
                batch_tensors = blur_transform(batch_tensors)

        return batch_tensors

    def process_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Process a batch of images efficiently"""
        # Step 1: Smart crop all images
        cropped_images = [self.smart_crop(img) for img in images]

        # Step 2: Convert to tensors and stack into batch
        if self.use_super_image:
            tensors = [IL.load_image(img).to(self.device) for img in cropped_images]
            batch_tensor = torch.cat(tensors, dim=0)
        else:
            # Fallback tensor conversion
            transform = torchvision.transforms.ToTensor()
            tensors = [transform(img).unsqueeze(0) for img in cropped_images]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)

        with torch.no_grad():
            # Step 3: Apply super-resolution or scaling to batch
            if self.use_super_image and self.config.get('method') == 'up':
                # Super-resolution model handles upsampling
                processed = model(batch_tensor)
                # Final resize with blur (super-resolution may need smoothing)
                target_size = self.config.get('target_size', (320, 240))
                processed = self.process_tensor(processed, target_size, skip_blur=False)
            else:
                # Regular scaling with proper anti-aliasing
                processed = self.scale_tensor(
                    batch_tensor,
                    self.config.get('method', 'up'),
                    self.config.get('upsample_factor', 2)
                )
                # Final resize without additional blur (already handled in scale_tensor)
                target_size = self.config.get('target_size', (320, 240))
                processed = self.process_tensor(processed, target_size, skip_blur=True)

            # Move to CPU for saving
            result = processed.cpu()

            # Clean up GPU memory
            del batch_tensor, processed
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            return result


def copy_metadata_files(input_root: str, output_root: str):
    """Copy trajectory and metadata files preserving folder structure"""
    metadata_files = ("traj_data.pkl", "traj_data.json", "camera_info.json")

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if filename in metadata_files:
                rel_dir = os.path.relpath(dirpath, input_root)
                dest_dir = os.path.join(output_root, rel_dir)
                os.makedirs(dest_dir, exist_ok=True)

                src = os.path.join(dirpath, filename)
                dst = os.path.join(dest_dir, filename)

                # Skip if already exists and same size
                if os.path.exists(dst) and os.path.getsize(src) == os.path.getsize(dst):
                    continue

                shutil.copy2(src, dst)
                print(f"Copied metadata: {filename}")

class SimplifiedBatchProcessor:
    """Simplified batch processor with automated configuration and progress logging"""

    def __init__(self, input_dir: str, output_dir: str, config: Optional[Dict[str, Any]] = None,
                 target_size: Optional[Tuple[int, int]] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size or (320, 240)

        # Initialize progress logger
        self.progress_logger = ProgressLogger(self.output_dir, self.input_dir)

        # Auto-detect configuration if not provided
        if config is None:
            config = auto_config.analyze_first_image(str(self.input_dir), self.target_size)
        elif not config.get('inferred_from_image', False):
            # If config provided but not inferred from image, still analyze for better parameters
            analyzed_config = auto_config.analyze_first_image(str(self.input_dir), self.target_size)
            # Merge analyzed parameters with provided config, giving priority to provided config
            for key in ['crop_pixels', 'upsample_factor', 'method']:
                if key not in config:
                    config[key] = analyzed_config.get(key, config.get(key))

        # Ensure target_size is set in config
        config['target_size'] = self.target_size

        self.config = config
        self.processor = ImageProcessor(config)
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

        print(f"🎯 Target size: {self.target_size}")
        print(f"⚙️  Config: crop={config.get('crop_pixels')}px, "
              f"factor={config.get('upsample_factor')}, method={config.get('method')}")

        # Check if we can resume from previous progress
        resume_info = self.progress_logger.get_resume_info()
        if resume_info['can_resume']:
            print(f"🔄 {resume_info['message']}")
        else:
            print(f"🆕 {resume_info['message']}")

    def get_image_folders(self) -> list:
        """Find all folders containing images"""
        print("🔍 Scanning for image folders...")
        folders = []

        # Get all directories first for progress tracking
        all_dirs = [d for d in self.input_dir.rglob('*') if d.is_dir()]

        # Progress bar for folder scanning
        scan_pbar = tqdm(all_dirs, desc="Scanning directories", unit="dir", leave=False)

        for dirpath in scan_pbar:
            # Check if folder contains images
            image_files = [f for f in dirpath.iterdir()
                         if f.is_file() and f.suffix.lower() in self.image_extensions]
            if image_files:
                folders.append(dirpath)
                scan_pbar.set_postfix({'found': len(folders)})

        scan_pbar.close()
        print(f"📁 Found {len(folders)} folders with images")
        return sorted(folders)



    def process_folder_batch(self, folder_path: Path, output_path: Path):
        """Batch processing using PyTorch DataLoader for efficient processing with progress tracking"""
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = [f for f in folder_path.iterdir()
                      if f.is_file() and f.suffix.lower() in self.image_extensions]

        if not image_files:
            print(f"No images found in {folder_path}")
            return

        image_files = sorted(image_files)

        # Check which images need processing (respecting overwrite flag)
        overwrite = self.config.get('overwrite', False)
        remaining_images = []

        if overwrite:
            # Process all images when overwrite is enabled
            remaining_images = image_files
            print(f"🔄 Overwrite mode: processing all {len(image_files)} images")
        else:
            # Only process images that don't exist (resumption mode)
            for img_file in image_files:
                output_file = output_path / (img_file.stem + '.jpg')
                if not output_file.exists():
                    remaining_images.append(img_file)

        if not remaining_images:
            if overwrite:
                print(f"✅ No images found in {folder_path.relative_to(self.input_dir)}")
            else:
                print(f"✅ All images in {folder_path.relative_to(self.input_dir)} already processed")
            return

        # Log progress start
        self.progress_logger.start_folder(folder_path, len(remaining_images))

        if not overwrite and len(remaining_images) < len(image_files):
            skipped = len(image_files) - len(remaining_images)
            print(f"📋 Resuming folder: {skipped} images already processed, {len(remaining_images)} remaining")

        # Create dataset and dataloader for remaining images only
        dataset = ImageDataset(remaining_images, self.processor)
        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 4)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep order for consistent output
            num_workers=num_workers,
            pin_memory=False,  # Disabled for memory efficiency
            drop_last=False,
            prefetch_factor=1 if num_workers > 0 else None,  # Only set if using workers
            persistent_workers=False  # Disabled for memory efficiency
        )

        processed_count = 0

        # Create progress bar for remaining images in this folder
        image_pbar = tqdm(total=len(remaining_images), desc="Processing images", unit="img",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} images [{elapsed}<{remaining}, {rate_fmt}]",
                         leave=False)

        for batch_idx, (batch_tensors, batch_paths) in enumerate(dataloader):
            try:
                # batch_tensors shape: [batch_size, 3, H, W] - no unnecessary reshaping

                # TRUE BATCH PROCESSING: Process entire batch at once
                processed_batch = self.processor.process_tensor_batch(batch_tensors)

                # OPTIMIZED SAVING: Save batch efficiently without unnecessary tensor operations
                self._save_batch_as_images(processed_batch, batch_paths, output_path)

                batch_size_actual = len(batch_paths)
                processed_count += batch_size_actual

                # Update progress bar
                image_pbar.update(batch_size_actual)
                image_pbar.set_postfix({
                    'batch': f"{batch_idx + 1}/{len(dataloader)}",
                    'batch_size': batch_size_actual
                })

                # Update progress logger every 10 batches or at the end
                if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                    self.progress_logger.update_folder_progress(processed_count)

                # Cleanup after each batch
                del batch_tensors, processed_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force garbage collection every 5 batches to prevent memory accumulation
                if batch_idx % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

            except Exception as e:
                print(f"\n❌ Error processing batch {batch_idx}: {e}")
                # Skip failed images in batch
                #for img_path in batch_paths:
                #    print(f"  Skipped: {img_path}")

                # Cleanup on error too
                if 'batch_tensors' in locals():
                    del batch_tensors
                if 'processed_batch' in locals():
                    del processed_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        image_pbar.close()

        # Complete folder in progress logger
        self.progress_logger.complete_folder(folder_path, processed_count)

        print(f"✅ Processed {processed_count} images")

    def _save_batch_as_images(self, batch_tensors: torch.Tensor, batch_paths: List[str], output_path: Path):
        """Efficiently save a batch of tensors as JPG images"""
        # batch_tensors shape: [batch_size, 3, H, W]
        for i, (tensor, img_path_str) in enumerate(zip(batch_tensors, batch_paths)):
            # Convert string path to Path object for processing
            img_path = Path(img_path_str)
            # Change extension to .jpg regardless of input format
            output_file = output_path / (img_path.stem + '.jpg')

            # Save tensor directly without unnecessary reshaping
            self._save_single_tensor_as_image(tensor, output_file)

            # Explicit cleanup for each tensor to prevent accumulation
            del tensor

    def _save_single_tensor_as_image(self, tensor: torch.Tensor, output_path: Path):
        """Save a single tensor as JPG image with maximum quality"""
        # tensor shape: [3, H, W] - no batch dimension to remove
        tensor_clamped = torch.clamp(tensor, 0, 1)  # Ensure valid range

        transform = torchvision.transforms.ToPILImage()
        img = transform(tensor_clamped)

        # Convert to RGB if not already (JPG doesn't support transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save as JPG with maximum quality (100)
        img.save(output_path, 'JPEG', quality=100)

        # Explicit cleanup
        del tensor_clamped, img

    def _save_tensor_as_image(self, tensor: torch.Tensor, output_path: Path):
        """Save tensor as JPG image with maximum quality"""
        # Convert tensor to PIL Image
        tensor = tensor.squeeze(0)  # Remove batch dimension
        tensor = torch.clamp(tensor, 0, 1)  # Ensure valid range

        transform = torchvision.transforms.ToPILImage()
        img = transform(tensor)

        # Convert to RGB if not already (JPG doesn't support transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save as JPG with maximum quality (100)
        img.save(output_path, 'JPEG', quality=100)

    def process_all_folders(self):
        """Main processing method using PyTorch DataLoader for batch processing with resumption support"""
        start_time = time.time()

        folders = self.get_image_folders()

        # Set total folders in progress logger
        self.progress_logger.set_total_folders(len(folders))

        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 4)
        print(f"⚙️  Using batch_size={batch_size}, num_workers={num_workers}")

        # Start processing

        # Filter folders based on overwrite flag
        overwrite = self.config.get('overwrite', False)
        remaining_folders = []

        if overwrite:
            # Process all folders when overwrite is enabled
            remaining_folders = folders
            print(f"🔄 Overwrite mode: processing all {len(folders)} folders")
        else:
            # Only process folders that aren't completed (resumption mode)
            for folder in folders:
                if not self.progress_logger.should_skip_folder(folder):
                    remaining_folders.append(folder)

            if len(remaining_folders) < len(folders):
                skipped = len(folders) - len(remaining_folders)
                print(f"📋 Resuming processing: {skipped} folders already completed, {len(remaining_folders)} remaining")

        # Count total images for speed calculation (only remaining folders)
        total_images = 0
        for folder in remaining_folders:
            image_files = [f for f in folder.iterdir()
                         if f.is_file() and f.suffix.lower() in self.image_extensions]
            total_images += len(image_files)

        if not remaining_folders:
            print("✅ All folders already processed!")
            return

        # Create overall progress bar for remaining folders
        folder_pbar = tqdm(remaining_folders, desc="Processing folders", unit="folder",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folders [{elapsed}<{remaining}]")

        try:
            for folder in folder_pbar:
                rel_path = folder.relative_to(self.input_dir)
                output_folder = self.output_dir / rel_path

                # Update folder progress bar description
                folder_pbar.set_description(f"Processing: {rel_path}")

                # Process the folder

                self.process_folder_batch(folder, output_folder)

                # Cleanup after each folder to prevent accumulation
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print(f"\n🛑 Processing interrupted by user. Progress saved.")
            print(f"📋 Resume by running the same command again.")
            raise
        except Exception as e:
            print(f"\n❌ Processing failed with error: {e}")
            print(f"📋 Progress saved. You can resume by running the same command again.")
            raise
        finally:
            # Ensure progress logger cleanup
            self.progress_logger.cleanup()

        folder_pbar.close()

        # Calculate processing statistics
        end_time = time.time()
        total_time = end_time - start_time
        images_per_second = total_images / total_time if total_time > 0 else 0

        print(f"\n🎉 Processing complete!")
        print(f"📊 Summary: {len(remaining_folders)} folders processed in this session")
        print(f"📊 Total: {len(folders)} folders, {total_images} images processed")
        print(f"⏱️  Time: {total_time:.1f}s ({images_per_second:.1f} images/sec)")
        print(f"💾 Output saved to: {self.output_dir}")
        print(f"📋 Progress log: {self.progress_logger.log_file}")

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset preprocessing with automatic parameter detection and metadata copying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (auto-detects parameters, copies metadata)
  python dataset_preprocess.py -i /path/to/input -o /path/to/output

  # Use config file with directories specified in config
  python dataset_preprocess.py --config config/data/preprocess_config.yaml

  # Use config file with command line directory overrides
  python dataset_preprocess.py --config config/data/preprocess_config.yaml -i /path/to/input -o /path/to/output

  # Override batch processing settings
  python dataset_preprocess.py -i /path/to/input -o /path/to/output --batch-size 16 --num-workers 8

  # Override image size
  python dataset_preprocess.py -i /path/to/input -o /path/to/output --width 640 --height 480

  # Overwrite existing processed images (instead of skipping them)
  python dataset_preprocess.py -i /path/to/input -o /path/to/output --overwrite

  # Crop only from top (useful for removing sky/ceiling from images)
  python dataset_preprocess.py -i /path/to/input -o /path/to/output --crop-from-top
        """
    )

    parser.add_argument("--input-dir", "-i", help="Input directory containing images (can be set in config)")
    parser.add_argument("--output-dir", "-o", help="Output directory for processed images (can be set in config)")
    parser.add_argument("--config", "-c", help="Configuration file (YAML or JSON)")

    # Override options
    parser.add_argument("--width", type=int, help="Override target width")
    parser.add_argument("--height", type=int, help="Override target height")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing (default: 8)")
    parser.add_argument("--num-workers", type=int, help="Number of worker processes for data loading (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed images (default: skip existing)")
    parser.add_argument("--crop-from-top", action="store_true", help="Crop only from top instead of evenly from top and bottom")


    args = parser.parse_args()

    # Load configuration
    config = None
    if args.config:
        config = load_config_file(args.config)
    else:
        # Use default configuration
        config = get_default_config()

    # Determine input and output directories
    input_dir = args.input_dir or config.get('input_dir')
    output_dir = args.output_dir or config.get('output_dir')

    # Validate required directories
    if not input_dir or not output_dir:
        parser.error("--input-dir and --output-dir are required (can be provided via command line or config file)")

    # Determine target size (with manual overrides)
    target_size = (320, 240)  # Default
    if config and 'target_size' in config:
        target_size = tuple(config['target_size'])
    if args.width or args.height:
        width = args.width or target_size[0]
        height = args.height or target_size[1]
        target_size = (width, height)
        if config:
            config['target_size'] = target_size

    # Apply batch processing overrides
    if config is None:
        config = {}
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_workers:
        config['num_workers'] = args.num_workers
    if args.overwrite:
        config['overwrite'] = args.overwrite
    if args.crop_from_top:
        config['crop_from_top'] = args.crop_from_top

    # Create processor and run
    processor = SimplifiedBatchProcessor(input_dir, output_dir, config, target_size)
    processor.process_all_folders()

    # Copy metadata files
    copy_metadata_files(input_dir, output_dir)

    print("\n✅ Processing completed successfully!")