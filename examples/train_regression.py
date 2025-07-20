"""Example training script for regression tasks with DuET."""

import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from duet import DualPatchTransformer, TimeSeriesDataModule


def main():
    """Run example training with synthetic regression data."""
    print("Creating synthetic regression dataset...")

    # Create synthetic data module for regression
    dm = TimeSeriesDataModule(
        batch_size=64,
        synthetic=True,
        num_samples=5000,
        T=100,  # Longer sequences
        C_num=8,  # 8 numeric features
        C_cat=2,  # 2 categorical features
        cat_cardinalities=[5, 3],
        task="regression",  # Regression task
        missing_ratio=0.1,
    )

    # Create model for regression
    model = DualPatchTransformer(
        C_num=8,
        C_cat=2,
        cat_cardinalities=[5, 3],
        T=100,
        d_model=64,
        task="regression",  # Regression task
        n_head=4,
        num_layers=3,
        lr=1e-3,
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/regression",
        filename="duet-reg-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min")

    # Logger with descriptive name
    logger = TensorBoardLogger(
        save_dir="logs",
        name="synthetic_regression",
        version=None,  # Auto-increment version
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint, early_stop],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # Train
    print("Starting regression training...")
    trainer.fit(model, dm)

    print("\nTraining complete!")
    print(f"Best model saved to: {checkpoint.best_model_path}")
    print(f"TensorBoard logs saved to: {logger.log_dir}")
    print("To view logs, run: tensorboard --logdir logs")


if __name__ == "__main__":
    main()
