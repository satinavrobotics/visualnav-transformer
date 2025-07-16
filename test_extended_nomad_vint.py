#!/usr/bin/env python3
"""
Test script for the extended NoMaD ViNT architecture with different encoder types.
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualnav_transformer.train.vint_train.models.nomad_vint import NoMaD_ViNT


def test_goal_image_encoder():
    """Test the original goal image encoder"""
    print("Testing goal_image encoder...")
    
    model = NoMaD_ViNT(
        context_size=3,
        obs_encoder="dinov2-small",
        obs_encoding_size=256,
        goal_encoder_type="goal_image"
    )
    
    # Create dummy inputs
    batch_size = 4
    obs_img = torch.randn(batch_size, 12, 96, 96)  # 4 context frames * 3 channels
    goal_img = torch.randn(batch_size, 3, 96, 96)
    
    # Forward pass
    output = model(obs_img, goal_img)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {output.shape}"
    print("✓ goal_image encoder test passed!")


def test_image_pair_encoder():
    """Test the image pair encoder"""
    print("\nTesting image_pair encoder...")
    
    model = NoMaD_ViNT(
        context_size=3,
        obs_encoder="dinov2-small",
        obs_encoding_size=256,
        goal_encoder_type="image_pair"
    )
    
    # Create dummy inputs
    batch_size = 4
    obs_img = torch.randn(batch_size, 12, 96, 96)  # 4 context frames * 3 channels
    goal_img = torch.randn(batch_size, 3, 96, 96)
    
    # Forward pass
    output = model(obs_img, goal_img)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {output.shape}"
    print("✓ image_pair encoder test passed!")


def test_goal_position_encoder():
    """Test the goal position encoder"""
    print("\nTesting goal_position encoder...")
    
    model = NoMaD_ViNT(
        context_size=3,
        obs_encoder="dinov2-small",
        obs_encoding_size=256,
        goal_encoder_type="goal_position"
    )
    
    # Create dummy inputs
    batch_size = 4
    obs_img = torch.randn(batch_size, 12, 96, 96)  # 4 context frames * 3 channels
    goal_pos = torch.randn(batch_size, 2)  # x, y coordinates
    
    # Forward pass
    output = model(obs_img, goal_pos=goal_pos)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {output.shape}"
    print("✓ goal_position encoder test passed!")


def test_image_pair_position_encoder():
    """Test the image pair + position encoder"""
    print("\nTesting image_pair_position encoder...")
    
    model = NoMaD_ViNT(
        context_size=3,
        obs_encoder="dinov2-small",
        obs_encoding_size=256,
        goal_encoder_type="image_pair_position"
    )
    
    # Create dummy inputs
    batch_size = 4
    obs_img = torch.randn(batch_size, 12, 96, 96)  # 4 context frames * 3 channels
    goal_img = torch.randn(batch_size, 3, 96, 96)
    goal_pos = torch.randn(batch_size, 2)  # x, y coordinates
    
    # Forward pass
    output = model(obs_img, goal_img, goal_pos)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {output.shape}"
    print("✓ image_pair_position encoder test passed!")


def test_prebuilt_features():
    """Test with prebuilt features"""
    print("\nTesting with prebuilt features...")
    
    model = NoMaD_ViNT(
        context_size=3,
        obs_encoder="dinov2-small",
        obs_encoding_size=256,
        goal_encoder_type="goal_position",
        use_prebuilt_features=True
    )
    
    # Create dummy inputs (prebuilt features)
    batch_size = 4
    obs_features = torch.randn(batch_size, 4, 384)  # context_size+1, feature_dim
    goal_pos = torch.randn(batch_size, 2)  # x, y coordinates
    
    # Forward pass
    output = model(obs_features, goal_pos=goal_pos)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {output.shape}"
    print("✓ prebuilt features test passed!")


def test_error_handling():
    """Test error handling for missing inputs"""
    print("\nTesting error handling...")
    
    model = NoMaD_ViNT(
        context_size=3,
        obs_encoder="dinov2-small",
        obs_encoding_size=256,
        goal_encoder_type="goal_position"
    )
    
    batch_size = 4
    obs_img = torch.randn(batch_size, 12, 96, 96)
    
    try:
        # This should raise an error because goal_pos is missing
        output = model(obs_img)
        assert False, "Expected ValueError for missing goal_pos"
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test invalid encoder type
    try:
        invalid_model = NoMaD_ViNT(goal_encoder_type="invalid_type")
        assert False, "Expected ValueError for invalid encoder type"
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")


if __name__ == "__main__":
    print("Testing Extended NoMaD ViNT Architecture")
    print("=" * 50)
    
    test_goal_image_encoder()
    test_image_pair_encoder()
    test_goal_position_encoder()
    test_image_pair_position_encoder()
    test_prebuilt_features()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("All tests passed! 🎉")
    print("\nThe extended NoMaD ViNT architecture supports:")
    print("1. goal_image: Original goal image encoding")
    print("2. image_pair: Current + goal image pair encoding")
    print("3. goal_position: Goal position (x, y) encoding")
    print("4. image_pair_position: Image pair + position encoding")
