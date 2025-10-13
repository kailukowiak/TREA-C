"""
Training script for W3 dataset with real wells only.

Usage:
    uv run python train_w3.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from treac.models import TriplePatchTransformer


class W3Dataset(Dataset):
    """Dataset for W3 well data."""

    def __init__(
        self,
        parquet_path: str,
        seq_len: int = 96,
        filter_simulated: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            parquet_path: Path to parquet file
            seq_len: Sequence length for time series windows
            filter_simulated: If True, exclude SIMULATED wells
            max_samples: Maximum number of samples to load (for memory management)
        """
        self.seq_len = seq_len

        # Load data
        print(f"Loading data from {parquet_path}...")
        if filter_simulated:
            # Load in chunks to filter efficiently
            df = pd.read_parquet(parquet_path)
            print(f"Original size: {len(df):,} rows")
            df = df[df['well_name'] != 'SIMULATED']
            print(f"After filtering SIMULATED: {len(df):,} rows")
        else:
            df = pd.read_parquet(parquet_path)

        # Limit samples if specified
        if max_samples is not None and len(df) > max_samples:
            print(f"Limiting to {max_samples:,} samples")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Feature columns (27 numeric features)
        self.feature_cols = [
            'ABER-CKGL', 'ABER-CKP', 'ESTADO-DHSV', 'ESTADO-M1', 'ESTADO-M2',
            'ESTADO-PXO', 'ESTADO-SDV-GL', 'ESTADO-SDV-P', 'ESTADO-W1', 'ESTADO-W2',
            'ESTADO-XO', 'P-ANULAR', 'P-JUS-BS', 'P-JUS-CKGL', 'P-JUS-CKP',
            'P-MON-CKGL', 'P-MON-CKP', 'P-MON-SDV-P', 'P-PDG', 'PT-P', 'P-TPT',
            'QBS', 'QGL', 'T-JUS-CKP', 'T-MON-CKP', 'T-PDG', 'T-TPT'
        ]

        # Extract features and labels
        self.features = df[self.feature_cols].values.astype(np.float32)
        self.labels = df['class'].values

        # Handle missing labels
        self.valid_indices = ~pd.isna(self.labels)
        self.features = self.features[self.valid_indices]
        self.labels = self.labels[self.valid_indices].astype(np.int64)

        print(f"Final dataset size: {len(self.labels):,} samples")
        print(f"Features shape: {self.features.shape}")
        print(f"Class distribution: {np.bincount(self.labels)}")

        # Compute normalization statistics (on training data only)
        self.mean = np.nanmean(self.features, axis=0)
        self.std = np.nanstd(self.features, axis=0)
        self.std[self.std == 0] = 1.0  # Avoid division by zero

    def __len__(self):
        # Number of possible windows
        return max(0, len(self.features) - self.seq_len + 1)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract sequence window
        seq = self.features[idx:idx + self.seq_len]

        # Normalize
        seq_norm = (seq - self.mean) / self.std

        # Replace NaNs with 0 (the model handles NaNs through mask channels)
        seq_norm = np.nan_to_num(seq_norm, nan=0.0)

        # Label for the window (use label at the end of sequence)
        label = self.labels[idx + self.seq_len - 1]

        return torch.from_numpy(seq_norm), torch.tensor(label, dtype=torch.long)


class W3DataModule(pl.LightningDataModule):
    """DataModule for W3 dataset."""

    def __init__(
        self,
        train_path: str = "./data/W3/train.parquet",
        test_path: str = "./data/W3/test.parquet",
        seq_len: int = 96,
        batch_size: int = 256,
        num_workers: int = 4,
        max_train_samples: Optional[int] = None,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Load full training data (filtered for real wells)
            full_train = W3Dataset(
                self.train_path,
                seq_len=self.seq_len,
                filter_simulated=True,
                max_samples=self.max_train_samples
            )

            # Split into train and validation
            train_size = int(len(full_train) * (1 - self.val_split))
            val_size = len(full_train) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            print(f"Train size: {len(self.train_dataset):,}")
            print(f"Val size: {len(self.val_dataset):,}")

        if stage == "test" or stage is None:
            self.test_dataset = W3Dataset(
                self.test_path,
                seq_len=self.seq_len,
                filter_simulated=False  # Test set doesn't have SIMULATED
            )
            print(f"Test size: {len(self.test_dataset):,}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )


class W3Model(pl.LightningModule):
    """Lightning wrapper for W3 classification."""

    def __init__(
        self,
        c_in: int = 27,
        seq_len: int = 96,
        num_classes: int = 17,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = TriplePatchTransformer(
            C_num=c_in,
            C_cat=0,  # No categorical features
            cat_cardinalities=[],
            T=seq_len,
            d_model=d_model,
            task='classification',
            num_classes=num_classes,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            lr=lr,
        )

        # Loss function (with class weights for imbalanced data)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


def main():
    # Set random seeds
    pl.seed_everything(42)

    # Configuration
    CONFIG = {
        'seq_len': 96,
        'batch_size': 256,
        'num_workers': 8,
        'max_train_samples': 5_000_000,  # Limit to 5M samples (adjust as needed)
        'val_split': 0.1,
        'max_epochs': 50,
        'd_model': 128,
        'n_head': 8,
        'num_layers': 3,
        'dropout': 0.1,
        'pooling': 'mean',
        'lr': 1e-4,
    }

    print("=" * 80)
    print("W3 Well Classification Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, val in CONFIG.items():
        print(f"  {key}: {val}")
    print()

    # Create data module
    datamodule = W3DataModule(
        seq_len=CONFIG['seq_len'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        max_train_samples=CONFIG['max_train_samples'],
        val_split=CONFIG['val_split'],
    )

    # Create model
    model = W3Model(
        c_in=27,
        seq_len=CONFIG['seq_len'],
        num_classes=17,  # Classes: 0-9, 101-109
        d_model=CONFIG['d_model'],
        n_head=CONFIG['n_head'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        pooling=CONFIG['pooling'],
        lr=CONFIG['lr'],
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/w3',
        filename='w3-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True,
    )

    # Logger
    logger = TensorBoardLogger('logs', name='w3_classification')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['max_epochs'],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        precision='16-mixed',  # Use mixed precision for faster training
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule)

    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, datamodule, ckpt_path='best')

    print("\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
