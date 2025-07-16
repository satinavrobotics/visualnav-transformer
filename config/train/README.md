# VisuaNav Transformer Training Configurations

This directory contains simplified training configurations for the VisuaNav Transformer model. The configurations have been streamlined to focus on the two main training scenarios.

## Available Configurations

### 1. Standard Configuration (`standard.yaml`)
- **Purpose**: Standard training with EfficientNet-B0 encoder
- **Key Overrides**:
  - `obs_encoder: efficientnet-b0` - Uses EfficientNet for feature extraction
  - `prebuilt_dino: False` - Processes images directly (no prebuilt features)
  - `goal_encoder_type: image_pair` - Inherited from defaults (processes image pairs)
  - `batch_size: 256` - Higher batch size for standard training
  - `num_workers: 12` - More workers for data loading
  - `warmup: True` - Enables learning rate warmup
  - `clipping: True` - Enables gradient clipping with range [-1.0, 1.0]
- **Inherits**: All other settings from `defaults.yaml`

### 2. Prebuilt DINO Features (`prebuilt_dino.yaml`)
- **Purpose**: Training with prebuilt DiNOv2 features for faster training
- **Key Overrides**:
  - `obs_encoder: dinov2-large` - Uses DiNOv2-Large for feature extraction
  - `prebuilt_dino: True` - Uses precomputed DINO features
  - `goal_encoder_type: feature_pair` - Processes prebuilt feature pairs
  - `batch_size: 64` - Reduced for memory efficiency
  - `num_workers: 0` - Prevents memory issues with prebuilt features
  - `image_size: [320, 240]` - Higher resolution for DINO features
  - `warmup: True` - Enables learning rate warmup
  - `clipping: True` - Enables gradient clipping with range [-0.5, 0.5] (more conservative)
- **Inherits**: All other settings from `defaults.yaml`

### 3. Default Configuration (`defaults.yaml`)
- **Purpose**: Base configuration with fallback values
- **Usage**: Automatically loaded first, then overridden by specific configs

## Key Parameters Explained

### Model Architecture
- `model_type: nomad` - Uses the NoMaD model architecture
- `vision_encoder: nomad_vint` - Vision encoder type
- `encoding_size: 256` - Feature encoding dimension
- `obs_encoder` - Observation encoder type (`efficientnet-b0` or `dinov2-large`)
- `prebuilt_dino` - Whether to use prebuilt DINO features
- `goal_encoder_type` - Type of goal encoding (depends on prebuilt_dino setting)

### Training Settings
- `train` - Enable/disable training (default: True, set to False for evaluation-only)
- `batch_size` - Training batch size
- `epochs` - Number of training epochs
- `lr` - Learning rate
- `optimizer` - Optimizer type (adamw recommended)
- `scheduler` - Learning rate scheduler
- `warmup` - Whether to use learning rate warmup

### Gradient Clipping
- `clipping` - Enable/disable gradient clipping (default: False)
- `min_norm` - Minimum gradient value for clipping (default: -1.0)
- `max_norm` - Maximum gradient value for clipping (default: 1.0)

**Gradient Clipping Strategy:**
- **Standard training**: Uses [-1.0, 1.0] range for robust training with EfficientNet
- **Prebuilt DINO**: Uses [-0.5, 0.5] range (more conservative) since features are pre-computed
- Clipping helps prevent gradient explosion and stabilizes training

### Diffusion Parameters
- `num_diffusion_iters` - Number of diffusion denoising steps
- `goal_mask_prob` - Probability of masking goal during training

### Context and Attention
- `context_size` - Number of context frames
- `mha_num_attention_heads` - Multi-head attention heads
- `mha_num_attention_layers` - Number of attention layers

### Goal Encoder Types
The `goal_encoder_type` parameter determines how goal information is processed:

**For Standard Training (prebuilt_dino: False):**
- `image_pair` - Processes current and goal images together (default)
- `position` - Uses goal position coordinates (x, y)
- `image_pair_position` - Combines image pairs with position information

**For Prebuilt DINO Features (prebuilt_dino: True):**
- `feature_pair` - Processes current and goal features together (default)
- `position` - Uses goal position coordinates (x, y)
- `feature_pair_position` - Combines feature pairs with position information

### GoalGMC Configuration
The `goal_gmc` section configures the Goal Multimodal Contrastive (GMC) module:

- `name` - Name identifier for the GoalGMC module
- `common_dim` - Common dimension for feature processing (default: 64)
- `latent_dim` - Latent dimension for encodings (default: 64)
- `loss_type` - Contrastive loss type ("infonce" or "joints_as_negatives")
- `learnable_temperature` - Whether temperature is learnable parameter (default: false)
- `initial_temperature` - Initial temperature value for contrastive loss (default: 0.1)

You can also specify `goal_gmc_config_path` to load GoalGMC settings from an external config file.

## Usage

### Training with Standard Configuration
```bash
python train.py --config config/train/standard.yaml
```

### Training with Prebuilt DINO Features
```bash
python train.py --config config/train/prebuilt_dino.yaml
```

## Configuration Inheritance

The training script loads configurations in this order:
1. **`defaults.yaml`** (base configuration with all parameters)
2. **Specified config file** (overrides only specific parameters)
3. **Runtime parameters** (timestamps, paths, etc.)

This means:
- `defaults.yaml` contains ALL required parameters with sensible defaults
- Specific configs (`standard.yaml`, `prebuilt_dino.yaml`) only override what's different
- This keeps the specific configs minimal and focused on their unique settings

## Memory Considerations

- **Standard config**: Uses more GPU memory for feature extraction but supports higher batch sizes
- **Prebuilt DINO config**: Uses less GPU memory during training but requires precomputed features
- Adjust `batch_size` and `num_workers` based on your hardware capabilities

## Removed Parameters

The following unused parameters have been removed from the simplified configs:
- `train_subset`, `eval_batch_size` - Not used in current training loop
- `clipping`, `max_norm` - Gradient clipping disabled
- `cyclic_period`, `plateau_patience`, `plateau_factor` - Unused scheduler parameters
- `attn_unet` - Always False
- `context_type` - Always temporal
- `pairwise_test_freq` - Set to 0 (disabled)
- Various goal encoder types - Simplified to default behavior

## Adding Custom Parameters

To add custom parameters:
1. Add them to `defaults.yaml` for global defaults
2. Override in specific config files as needed
3. Ensure the training code supports the new parameters
