#!/usr/bin/env python3
"""Test pre-training with only temporal order prediction to isolate that component."""

import subprocess
import sys


def run_temporal_only():
    """Run pre-training with only temporal order prediction."""
    cmd = [
        "uv",
        "run",
        "scripts/pretrain.py",
        "--max_epochs",
        "1",
        "--batch_size",
        "4",
        "--d_model",
        "32",
        "--sequence_length",
        "64",
        "--num_layers",
        "1",
        "--n_head",
        "2",
        "--lr",
        "1e-5",
        "--no_masked_patch",
        "--no_contrastive",
        "--lambda_temporal",
        "0.1",
        "--lambda_supervised",
        "1.0",
    ]

    print("Running pre-training with ONLY temporal order prediction...")
    print("Command:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ Temporal-only pre-training completed!")

            # Check if there were NaN values in the output
            output = result.stdout + result.stderr
            if "nan" in output.lower() or "NaN" in output:
                print("⚠️  WARNING: Found NaN values in temporal order prediction!")
                # Extract relevant lines
                lines = output.split("\n")
                nan_lines = [
                    line for line in lines if "nan" in line.lower() or "NaN" in line
                ]
                for line in nan_lines[-5:]:  # Last 5 NaN-related lines
                    print(f"  {line}")
                return False
            else:
                print("✅ Temporal order prediction is stable!")
                return True
        else:
            print("❌ Temporal-only pre-training failed!")
            print("STDOUT:", result.stdout[-1000:])
            print("STDERR:", result.stderr[-1000:])
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Pre-training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running pre-training: {e}")
        return False


if __name__ == "__main__":
    success = run_temporal_only()
    sys.exit(0 if success else 1)
