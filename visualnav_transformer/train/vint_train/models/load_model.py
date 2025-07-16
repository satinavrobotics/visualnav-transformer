import torch

from visualnav_transformer.train.vint_train.models.nomad import (
    DenseNetwork,
    NoMaD,
    PoseHead
)
from visualnav_transformer.train.vint_train.models.nomad_vint import NoMaD_ViNT
from visualnav_transformer.train.vint_train.models.utils import replace_bn_with_gn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# Import CLI formatter for enhanced output
try:
    from visualnav_transformer.train.vint_train.logging.cli_formatter import (
        Colors, Symbols, print_info, print_success, print_section, format_number
    )
except ImportError:
    # Fallback if cli_formatter is not available
    def print_info(msg, symbol=None):
        print(f"INFO: {msg}")
    def print_success(msg, symbol=None):
        print(f"SUCCESS: {msg}")
    def print_section(title, symbol=None, color=None):
        print(f"\n=== {title} ===")
    def format_number(num, precision=0):
        return str(num)
    class Symbols:
        ROCKET = '🚀'
        INFO = 'ℹ️'
        SUCCESS = '✅'
    class Colors:
        BRIGHT_GREEN = '\033[92m'

def load_model(config):
    """Load and initialize the NoMaD model with enhanced CLI output."""
    print_section("Model Initialization", Symbols.ROCKET, Colors.BRIGHT_GREEN)

    use_prebuilt_features = config.get("prebuilt_dino", False)

    # Display model configuration
    print_info(f"Model type: NoMaD (Navigation with Diffusion)", Symbols.INFO)
    print_info(f"Encoding size: {config['encoding_size']}", Symbols.INFO)
    print_info(f"Context size: {config['context_size']}", Symbols.INFO)
    print_info(f"Observation encoder: {config['obs_encoder']}", Symbols.INFO)
    print_info(f"Using pre-built DINO features: {use_prebuilt_features}", Symbols.INFO)

    # Get GoalGMC configuration
    goal_gmc_config = config.get("goal_gmc", {
        "name": "goal_gmc",
        "common_dim": 64,
        "latent_dim": 64,
        "loss_type": "infonce",
        "learnable_temperature": False,
        "initial_temperature": 0.1
    })

    # Get GoalGMC weights path
    goal_gmc_weights_path = config.get("goal_gmc_weights_path", None)

    # Display goal encoding configuration
    goal_encoder_type = config.get("goal_encoder_type", "goal_image")
    print_info(f"Goal encoder type: {goal_encoder_type}", Symbols.INFO)
    if goal_encoder_type == "goal_gmc":
        print_info(f"GoalGMC common dim: {goal_gmc_config['common_dim']}", Symbols.INFO)
        print_info(f"GoalGMC loss type: {goal_gmc_config['loss_type']}", Symbols.INFO)

    # Create vision encoder
    print_info("Creating vision encoder (NoMaD_ViNT)...", Symbols.ROCKET)
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        obs_encoder=config["obs_encoder"],  # Pass the obs_encoder parameter
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        use_prebuilt_features=use_prebuilt_features,
        goal_encoder_type=config.get("goal_encoder_type", "goal_image"),  # Default to original behavior
        goal_gmc_config=goal_gmc_config,  # Pass GoalGMC configuration
        goal_gmc_weights_path=goal_gmc_weights_path,  # Pass GoalGMC weights path
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    print_success("Vision encoder created successfully", Symbols.SUCCESS)

    # Create diffusion components
    print_info("Creating diffusion noise prediction network...", Symbols.ROCKET)
    noise_pred_net = ConditionalUnet1D(
        input_dim=4,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    print_success("Diffusion network created successfully", Symbols.SUCCESS)

    # Create auxiliary networks
    print_info("Creating auxiliary prediction networks...", Symbols.ROCKET)
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    pose_head = PoseHead(input_dim=config["encoding_size"])
    print_success("Auxiliary networks created successfully", Symbols.SUCCESS)

    # Assemble complete model
    print_info("Assembling complete NoMaD model...", Symbols.ROCKET)
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
        pose_head=pose_head,
    )

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_success(f"Model assembled: {format_number(total_params)} total parameters", Symbols.SUCCESS)
    print_info(f"Trainable parameters: {format_number(trainable_params)}", Symbols.INFO)

    # Create noise scheduler
    print_info("Creating diffusion noise scheduler...", Symbols.ROCKET)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    print_info(f"Diffusion timesteps: {config['num_diffusion_iters']}", Symbols.INFO)
    print_info(f"Beta schedule: squaredcos_cap_v2", Symbols.INFO)
    print_success("Noise scheduler created successfully", Symbols.SUCCESS)

    # Apply gradient clipping if enabled
    if config.get("clipping", False):
        min_norm = config.get("min_norm", -1.0)
        max_norm = config.get("max_norm", 1.0)
        print_info(f"Applying gradient clipping: [{min_norm}, {max_norm}]", Symbols.INFO)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(grad, min_norm, max_norm)
            )
        print_success("Gradient clipping applied successfully", Symbols.SUCCESS)

    print_success("Model initialization completed!", Symbols.SUCCESS)
    print()
    return model, noise_scheduler
