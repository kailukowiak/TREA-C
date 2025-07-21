#!/usr/bin/env python3
"""Quick test script for debugging pre-training issues."""

import sys

import torch

from duet.models.pretrain_patch_duet import PretrainPatchDuET


def test_ssl_forward():
    """Test SSL forward pass with simple inputs."""
    print("Testing SSL forward pass...")

    # Create model with conservative settings
    model = PretrainPatchDuET(
        max_numeric_features=4,
        max_categorical_features=0,
        num_classes=3,
        ssl_objectives={
            "masked_patch": True,
            "temporal_order": True,
            "contrastive": True,
        },
        d_model=32,
        n_head=2,
        num_layers=1,
        patch_len=8,
        stride=4,
        lr=1e-4,
    )

    # Set schema
    model.set_dataset_schema(numeric_features=4, categorical_features=0)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test data
    B, C, T = 2, 4, 32
    x_num = torch.randn(B, C, T) * 0.1  # Small values
    feature_mask = torch.ones(B, C, T)

    # Add some controlled NaNs
    x_num[0, 0, :5] = float("nan")

    print(f"Input shape: {x_num.shape}")
    print(
        f"Input range: {x_num[~torch.isnan(x_num)].min():.3f} to {x_num[~torch.isnan(x_num)].max():.3f}"
    )

    try:
        # Test SSL forward
        ssl_outputs = model.forward_ssl(x_num, feature_mask)
        print(f"SSL outputs keys: {list(ssl_outputs.keys())}")

        # Test SSL losses
        ssl_losses = model.compute_ssl_losses(ssl_outputs)
        print(f"SSL losses: {[(k, v.item()) for k, v in ssl_losses.items()]}")

        # Check for NaN/Inf
        for key, loss in ssl_losses.items():
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: {key} loss is {loss}")
                return False

        # Test standard forward
        output = model(x_num)
        print(f"Classification output: {output.shape}")

        if torch.isnan(output).any() or torch.isinf(output).any():
            print("WARNING: Classification output contains NaN/Inf")
            return False

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_batch_training():
    """Test a single training step."""
    print("\nTesting batch training...")

    # Create smaller model
    model = PretrainPatchDuET(
        max_numeric_features=4,
        max_categorical_features=0,
        num_classes=3,
        ssl_objectives={
            "masked_patch": True,
            "temporal_order": False,  # Disable temporal for now
            "contrastive": False,  # Disable contrastive for now
        },
        d_model=32,
        n_head=2,
        num_layers=1,
        lr=1e-4,
        lambda_masked=0.1,
        lambda_supervised=1.0,
    )

    # Set schema
    model.set_dataset_schema(numeric_features=4, categorical_features=0)

    # Create batch
    B = 4
    batch = {
        "x_num": torch.randn(B, 4, 32) * 0.1,
        "x_cat": None,
        "y": torch.randint(0, 3, (B,)),
        "feature_mask": torch.ones(B, 4, 32),
        "dataset_name": ["test"] * B,
        "num_classes": torch.tensor([3] * B),
    }

    # Add some NaNs
    batch["x_num"][0, :2, :5] = float("nan")

    try:
        # Training step
        model.train()
        loss = model.training_step(batch, 0)

        print(f"Training loss: {loss.item()}")

        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ Training loss is NaN/Inf")
            return False

        print("✅ Training step successful!")
        return True

    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("PRE-TRAINING DEBUG TEST")
    print("=" * 50)

    success1 = test_ssl_forward()
    success2 = test_batch_training()

    if success1 and success2:
        print("\n🎉 All tests passed! Pre-training should work.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the issues above.")
        sys.exit(1)
