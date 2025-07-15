"""Example training script for DuET."""

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger

from duet import DualPatchTransformer, TimeSeriesDataModule


def main():
    """Run example training with synthetic data."""
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
        name="duet_experiment",
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
