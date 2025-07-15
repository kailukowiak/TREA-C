"""Example training script for DuET."""

import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.loggers import TensorBoardLogger

from duet import DualPatchTransformer, TimeSeriesDataModule
from duet.data.dataset import SyntheticTimeSeriesDataset


def main():
    """Run example training with synthetic data."""
    # --- Inspect Data ---
    print("Generating and inspecting synthetic data...")
    # Create a dataset instance to inspect the data
    inspect_dataset = SyntheticTimeSeriesDataset(
        num_samples=20,
        T=64,
        C_num=4,
        C_cat=3,
        cat_cardinalities=[7, 20, 5],
        task="classification",
        num_classes=3,
        missing_ratio=0.15,
    )

    # Prepare data for DataFrame
    data_for_df = {"label": [inspect_dataset[i]["y"].item() for i in range(20)]}
    # Add mean of each numeric channel
    for i in range(inspect_dataset.C_num):
        data_for_df[f"num_{i}_mean"] = [
            torch.nanmean(inspect_dataset[j]["x_num"][i]).item() for j in range(20)
        ]

    # Add first value of each categorical channel
    for i in range(inspect_dataset.C_cat):
        data_for_df[f"cat_{i}_first"] = [
            inspect_dataset[j]["x_cat"][i][0].item() for j in range(20)
        ]

    df = pd.DataFrame(data_for_df)
    print("--- Synthetic Data Head (20 samples) ---")
    print(df)
    print("----------------------------------------")

    # Create synthetic data module with variable categorical cardinalities
    dm = TimeSeriesDataModule(
        batch_size=32,
        synthetic=True,
        num_samples=1000,
        T=64,
        C_num=4,
        C_cat=3,
        cat_cardinalities=[7, 20, 5],  # Different cardinalities per feature
        task="classification",
        num_classes=3,
        missing_ratio=0.15,
    )

    # Create model
    model = DualPatchTransformer(
        C_num=4,
        C_cat=3,
        cat_cardinalities=[7, 20, 5],
        T=64,
        d_model=64,
        task="classification",
        num_classes=3,
        nhead=4,
        num_layers=2,
        lr=1e-3,
    )

    # Create TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name="synthetic_classification",
        version=None,  # Auto-increment version
        log_graph=True,
        default_hp_metric=False,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[],
    )

    # Train
    trainer.fit(model, dm)

    print("Training complete!")
    print(f"TensorBoard logs saved to: {logger.log_dir}")
    print("To view logs, run one of:")
    print("  tensorboard --logdir logs")
    print("  uv run scripts/tensorboard.py")


if __name__ == "__main__":
    main()
