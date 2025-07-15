"""Example training script for DuET."""

import pytorch_lightning as pl

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
        task='classification',
        num_classes=3,
        missing_ratio=0.15
    )
    
    # Create model
    model = DualPatchTransformer(
        C_num=4,
        C_cat=3,
        cat_cardinalities=[7, 20, 5],
        T=64,
        d_model=64,
        task='classification',
        num_classes=3,
        nhead=4,
        num_layers=2,
        lr=1e-3
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        logger=True,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(model, dm)
    
    print("Training complete!")


if __name__ == "__main__":
    main()