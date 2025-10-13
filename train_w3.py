"""
Training script for W3 dataset with real wells only.

Usage:
    uv run python train_w3.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

from treac.models import TriplePatchTransformer


class W3Dataset(Dataset):
    """Dataset for W3 well data."""

    def __init__(
        self,
        parquet_path: str,
        seq_len: int = 96,
        filter_simulated: bool = True,
        max_samples: Optional[int] = None,
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
            df = df[df["well_name"] != "SIMULATED"]
            print(f"After filtering SIMULATED: {len(df):,} rows")
        else:
            df = pd.read_parquet(parquet_path)

        # Limit samples if specified
        if max_samples is not None and len(df) > max_samples:
            print(f"Limiting to {max_samples:,} samples")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Feature columns (27 numeric features)
        self.feature_cols = [
            "ABER-CKGL",
            "ABER-CKP",
            "ESTADO-DHSV",
            "ESTADO-M1",
            "ESTADO-M2",
            "ESTADO-PXO",
            "ESTADO-SDV-GL",
            "ESTADO-SDV-P",
            "ESTADO-W1",
            "ESTADO-W2",
            "ESTADO-XO",
            "P-ANULAR",
            "P-JUS-BS",
            "P-JUS-CKGL",
            "P-JUS-CKP",
            "P-MON-CKGL",
            "P-MON-CKP",
            "P-MON-SDV-P",
            "P-PDG",
            "PT-P",
            "P-TPT",
            "QBS",
            "QGL",
            "T-JUS-CKP",
            "T-MON-CKP",
            "T-PDG",
            "T-TPT",
        ]

        # Extract features and labels
        # First, handle extreme values that cause overflow
        features_raw = df[self.feature_cols].values

        # Clip extreme values before converting to float32
        features_raw = np.clip(features_raw, -1e6, 1e6)
        self.features = features_raw.astype(np.float32)

        self.labels = df["class"].values

        # Handle missing labels
        self.valid_indices = ~pd.isna(self.labels)
        self.features = self.features[self.valid_indices]
        self.labels = self.labels[self.valid_indices].astype(np.int64)

        print(f"Final dataset size: {len(self.labels):,} samples")
        print(f"Features shape: {self.features.shape}")

        # Better class distribution printing for sparse labels
        unique_classes, counts = np.unique(self.labels, return_counts=True)
        print(f"\nOriginal class distribution:")
        for cls, count in zip(unique_classes, counts):
            print(f"  Class {cls}: {count:,} samples")

        # Create label mapping: classes 0-9 stay the same, 101-109 map to 10-18
        # This ensures labels are contiguous from 0 to num_classes-1
        self.label_mapping = {}
        remapped_idx = 0
        for cls in sorted(unique_classes):
            self.label_mapping[cls] = remapped_idx
            remapped_idx += 1

        # Apply label mapping
        self.labels = np.array(
            [self.label_mapping[label] for label in self.labels], dtype=np.int64
        )

        print(f"\nRemapped class distribution:")
        unique_remapped, counts_remapped = np.unique(self.labels, return_counts=True)
        for cls, orig_cls, count in zip(
            unique_remapped, sorted(unique_classes), counts_remapped
        ):
            print(f"  Class {cls} (originally {orig_cls}): {count:,} samples")

        self.num_classes = len(unique_classes)
        print(f"\nTotal unique classes: {self.num_classes}")

        # Compute class weights for imbalanced dataset
        # Use inverse frequency: weight = 1 / (class_count / total_count)
        total_samples = len(self.labels)
        self.class_weights = np.zeros(self.num_classes, dtype=np.float32)
        for remapped_cls, count in zip(unique_remapped, counts_remapped):
            self.class_weights[remapped_cls] = total_samples / (
                count * self.num_classes
            )

        print(f"\nClass weights (for loss function):")
        for cls, orig_cls, weight in zip(
            unique_remapped, sorted(unique_classes), self.class_weights
        ):
            print(f"  Class {cls} (originally {orig_cls}): weight = {weight:.4f}")

        # Compute normalization statistics (on training data only)
        # Handle columns that are all NaN
        self.mean = np.nanmean(self.features, axis=0)
        self.std = np.nanstd(self.features, axis=0)

        # Replace NaN mean/std with 0 and 1 respectively
        self.mean = np.nan_to_num(self.mean, nan=0.0)
        self.std = np.nan_to_num(self.std, nan=1.0)
        self.std[self.std == 0] = 1.0  # Avoid division by zero

    def __len__(self):
        # Number of possible windows
        return max(0, len(self.features) - self.seq_len + 1)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract sequence window
        seq = self.features[idx : idx + self.seq_len]

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
                max_samples=self.max_train_samples,
            )

            # Split into train and validation
            train_size = int(len(full_train) * (1 - self.val_split))
            val_size = len(full_train) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            print(f"Train size: {len(self.train_dataset):,}")
            print(f"Val size: {len(self.val_dataset):,}")

        if stage == "test" or stage is None:
            self.test_dataset = W3Dataset(
                self.test_path,
                seq_len=self.seq_len,
                filter_simulated=False,  # Test set doesn't have SIMULATED
            )
            print(f"Test size: {len(self.test_dataset):,}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )


class GradientNormCallback(Callback):
    """Callback to log gradient norms for debugging training dynamics."""

    def on_after_backward(self, trainer, pl_module):
        """Log gradient norms after backward pass."""
        # Calculate total gradient norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        # Log to tensorboard
        pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)


class ConfusionMatrixCallback(Callback):
    """Callback to log confusion matrix at the end of each validation epoch."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update confusion matrix with validation batch predictions."""
        x, y = batch
        logits = pl_module(x)
        preds = torch.argmax(logits, dim=1)
        self.confusion_matrix.update(preds, y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log confusion matrix as an image to TensorBoard."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Compute confusion matrix
        cm = self.confusion_matrix.compute().cpu().numpy()
        self.confusion_matrix.reset()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - Epoch {trainer.current_epoch}")

        # Log to tensorboard
        trainer.logger.experiment.add_figure(
            "confusion_matrix", fig, global_step=trainer.global_step
        )
        plt.close(fig)


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
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Create model
        self.model = TriplePatchTransformer(
            C_num=c_in,
            C_cat=0,  # No categorical features
            cat_cardinalities=[],
            T=seq_len,
            d_model=d_model,
            task="classification",
            num_classes=num_classes,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            lr=lr,
        )

        # Loss function with class weights for imbalanced data
        if class_weights is not None:
            print(f"\nUsing weighted loss with class weights")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            print(f"\nUsing unweighted loss")
            self.criterion = nn.CrossEntropyLoss()

        # Metrics for evaluation
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_per_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average="none")
        self.val_per_class_f1 = F1Score(task="multiclass", num_classes=num_classes, average="none")

        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_per_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average="none")
        self.test_per_class_f1 = F1Score(task="multiclass", num_classes=num_classes, average="none")

    def forward(self, x):
        # TriplePatchTransformer expects (x_num, x_cat)
        # x has shape [B, T, C], need to transpose to [B, C, T]
        x_num = x.transpose(1, 2)
        x_cat = torch.zeros(
            x.shape[0], 0, x.shape[1], dtype=torch.long, device=x.device
        )
        return self.model(x_num, x_cat)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_per_class_acc(preds, y)
        self.val_per_class_f1(preds, y)

        # Log macro metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc_macro", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val_f1_macro", self.val_f1, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Log per-class metrics at the end of validation epoch."""
        per_class_acc = self.val_per_class_acc.compute()
        per_class_f1 = self.val_per_class_f1.compute()

        # Log per-class metrics
        for i in range(len(per_class_acc)):
            self.log(f"val_acc_class_{i}", per_class_acc[i], on_epoch=True)
            self.log(f"val_f1_class_{i}", per_class_f1[i], on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_per_class_acc(preds, y)
        self.test_per_class_f1(preds, y)

        # Log macro metrics
        self.log("test_loss", loss)
        self.log("test_acc_macro", self.test_acc, on_epoch=True)
        self.log("test_f1_macro", self.test_f1, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """Log per-class metrics at the end of test epoch."""
        per_class_acc = self.test_per_class_acc.compute()
        per_class_f1 = self.test_per_class_f1.compute()

        # Log per-class metrics
        for i in range(len(per_class_acc)):
            self.log(f"test_acc_class_{i}", per_class_acc[i], on_epoch=True)
            self.log(f"test_f1_class_{i}", per_class_f1[i], on_epoch=True)

        # Print summary to console
        print("\nPer-class Test Results:")
        print("=" * 60)
        for i in range(len(per_class_acc)):
            print(f"  Class {i}: Acc={per_class_acc[i]:.4f}, F1={per_class_f1[i]:.4f}")
        print("=" * 60)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def main():
    # Set random seeds
    pl.seed_everything(42)

    # Set matmul precision for better performance on CUDA devices with Tensor Cores
    torch.set_float32_matmul_precision("medium")

    # Configuration
    CONFIG = {
        "seq_len": 96,
        "batch_size": 256,
        "num_workers": 8,
        "max_train_samples": 5_000_000,  # Limit to 5M samples (adjust as needed)
        "val_split": 0.1,
        "max_epochs": 50,
        "d_model": 128,
        "n_head": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "pooling": "mean",
        "lr": 1e-4,
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
        seq_len=CONFIG["seq_len"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        max_train_samples=CONFIG["max_train_samples"],
        val_split=CONFIG["val_split"],
    )

    # Setup to get the number of classes from the dataset
    datamodule.setup(stage="fit")

    # Get num_classes and class_weights from the dataset
    if hasattr(datamodule.train_dataset, "dataset"):
        # Handle RandomSplit wrapper
        train_dataset = datamodule.train_dataset.dataset
    else:
        train_dataset = datamodule.train_dataset

    num_classes = train_dataset.num_classes
    class_weights = torch.from_numpy(train_dataset.class_weights).float()

    print(f"\nDetected {num_classes} classes in dataset")

    # Create model
    model = W3Model(
        c_in=27,
        seq_len=CONFIG["seq_len"],
        num_classes=num_classes,
        d_model=CONFIG["d_model"],
        n_head=CONFIG["n_head"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        pooling=CONFIG["pooling"],
        lr=CONFIG["lr"],
        class_weights=class_weights,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/w3",
        filename="w3-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    confusion_matrix_callback = ConfusionMatrixCallback(num_classes=num_classes)
    gradient_norm_callback = GradientNormCallback()

    # Logger
    logger = TensorBoardLogger("logs", name="w3_classification")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator="auto",
        devices=1,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            confusion_matrix_callback,
            gradient_norm_callback,
        ],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        precision="16-mixed",  # Use mixed precision for faster training
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule)

    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, datamodule, ckpt_path="best")

    print("\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
