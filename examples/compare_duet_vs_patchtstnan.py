#!/usr/bin/env python3
"""
Compare DuET vs PatchTSTNan with NaN injection on ETTh1 dataset.
"""

import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score

from duet.data.datamodule_v2 import TimeSeriesDataModuleV2
from duet.data.etth1 import ETTh1Dataset
from duet.models.transformer import DualPatchTransformer


class PatchTSTNan(pl.LightningModule):
    """Custom PatchTST with dual-patch NaN handling from DuET."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding - NOTE: input size is 2*c_in due to dual-patch (value + mask)
        self.patch_embedding = nn.Linear(patch_len * (2 * c_in), d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, n_head=n_head, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.head = nn.Linear(d_model, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def create_patches(self, x):
        """Create patches from input tensor."""
        B, C, T = x.shape
        patches = []

        for i in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, :, i : i + self.patch_len]  # [B, C, patch_len]
            patch = patch.reshape(B, -1)  # [B, C * patch_len]
            patches.append(patch)

        patches = torch.stack(patches, dim=1)  # [B, num_patches, C * patch_len]
        return patches

    def forward(self, x_num, x_cat=None):
        # Apply dual-patch NaN handling from DuET
        m_nan = torch.isnan(x_num).float()  # [B, C_num, T]
        x_val = torch.nan_to_num(x_num, nan=0.0)  # [B, C_num, T]

        # Stack value & mask channels (dual-patch)
        x_num2 = torch.cat([x_val, m_nan], dim=1)  # [B, 2·C_num, T]

        # Create patches from dual-patch input
        patches = self.create_patches(x_num2)

        # Embed patches
        x = self.patch_embedding(patches)
        x = x + self.pos_embedding

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)  # [B, d_model]

        # Classification
        return self.head(x)

    def training_step(self, batch, _batch_idx):
        x = batch["x_num"]
        labels = batch["y"]

        # Ensure labels are in the right format
        if labels.ndim == 2:
            labels = torch.argmax(labels, dim=1)
        labels = labels.long()

        logits = self.forward(x)
        loss = self.loss_fn(logits, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x = batch["x_num"]
        labels = batch["y"]

        # Ensure labels are in the right format
        if labels.ndim == 2:
            labels = torch.argmax(labels, dim=1)
        labels = labels.long()

        logits = self.forward(x)
        loss = self.loss_fn(logits, labels)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def inject_nans(dataset, nan_rate=0.05):
    """Inject NaN values into the dataset at specified rate."""
    print(f"Injecting NaN values at rate: {nan_rate:.1%}")

    # Create a copy of the dataset
    dataset_copy = type(dataset)(
        data_dir=dataset.data_dir,
        train=dataset.train,
        sequence_length=dataset.sequence_length,
        task=dataset.task,
        num_classes=dataset.num_classes,
        download=False,
    )

    # Inject NaNs into x_num
    x_num = dataset_copy.x_num.clone()

    # Generate random mask for NaN injection
    nan_mask = torch.rand_like(x_num) < nan_rate
    x_num[nan_mask] = float("nan")

    # Update the dataset
    dataset_copy.x_num = x_num

    # Calculate actual NaN percentage
    total_values = x_num.numel()
    nan_count = torch.isnan(x_num).sum().item()
    actual_nan_rate = nan_count / total_values

    print(
        f"Actual NaN rate: {actual_nan_rate:.1%} "
        f"({nan_count:,} / {total_values:,} values)"
    )

    return dataset_copy


def train_model(model, dm, model_name, max_epochs=15):
    """Train a model and return results."""

    # Callbacks
    checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/nan_comparison/{model_name}",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True,
        logger=False,
    )

    # Train
    start_time = time.time()
    trainer.fit(model, dm)
    training_time = time.time() - start_time

    # Load best model for evaluation
    best_model = type(model).load_from_checkpoint(
        checkpoint.best_model_path, **model.hparams
    )
    best_model.eval()

    # Skip test step since we're using validation set for evaluation

    # Get predictions for metrics
    best_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dm.val_dataloader():
            x_num = batch["x_num"]
            x_cat = batch.get("x_cat", None)
            labels = batch["y"]

            # Move tensors to the same device as the model
            device = next(best_model.parameters()).device
            x_num = x_num.to(device)
            if x_cat is not None:
                x_cat = x_cat.to(device)
            labels = labels.to(device)

            if labels.ndim == 2:
                labels = torch.argmax(labels, dim=1)
            labels = labels.long()

            # Call forward with appropriate arguments based on model type
            if isinstance(best_model, DualPatchTransformer):
                logits = best_model(x_num, x_cat)
            else:
                logits = best_model(x_num)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # Get final validation loss
    val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
    if torch.is_tensor(val_loss):
        val_loss = val_loss.item()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Val Loss": val_loss,
        "Training Time (s)": training_time,
        "Parameters": sum(p.numel() for p in model.parameters()),
        "Epochs": trainer.current_epoch + 1,
    }


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    MAX_EPOCHS = 15
    NAN_RATE = 0.05  # 5% NaN injection

    print("=" * 80)
    print("DuET vs PatchTSTNan Comparison with NaN Injection")
    print("=" * 80)

    # Load clean dataset
    print("\nLoading ETTh1 dataset...")
    train_dataset = ETTh1Dataset(train=True)
    val_dataset = ETTh1Dataset(train=False)

    # Inject NaNs
    print(f"\nInjecting {NAN_RATE:.1%} NaN values...")
    train_dataset_nan = inject_nans(train_dataset, nan_rate=NAN_RATE)
    val_dataset_nan = inject_nans(val_dataset, nan_rate=NAN_RATE)

    # Create data module
    dm = TimeSeriesDataModuleV2(
        train_dataset=train_dataset_nan,
        val_dataset=val_dataset_nan,
        batch_size=64,
        num_workers=4,
    )

    # Get dataset info
    c_in = train_dataset_nan.x_num.shape[1]
    seq_len = train_dataset_nan.sequence_length
    num_classes = train_dataset_nan.num_classes

    print(f"Train samples: {len(train_dataset_nan)}")
    print(f"Val samples: {len(val_dataset_nan)}")
    print("\nDataset info:")
    print(f"- Input channels: {c_in}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of classes: {num_classes}")

    results = []

    # 1. Train DuET
    print("\n" + "=" * 60)
    print("Training DuET (Dual-Patch Transformer)...")
    print("=" * 60)

    duet_model = DualPatchTransformer(
        C_num=c_in,
        C_cat=0,  # No categorical features
        cat_cardinalities=[],
        T=seq_len,
        d_model=128,
        task="classification",
        num_classes=num_classes,
        n_head=8,
        num_layers=3,
        lr=1e-3,
        pooling="mean",
        dropout=0.1,
    )

    duet_results = train_model(duet_model, dm, "DuET", max_epochs=MAX_EPOCHS)
    results.append(duet_results)

    # 2. Train PatchTSTNan
    print("\n" + "=" * 60)
    print("Training PatchTSTNan...")
    print("=" * 60)

    patchtst_nan_model = PatchTSTNan(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        patch_len=16,
        stride=8,
        d_model=128,
        n_head=8,
        num_layers=3,
        lr=1e-3,
    )

    patchtst_nan_results = train_model(
        patchtst_nan_model, dm, "PatchTSTNan", max_epochs=MAX_EPOCHS
    )
    results.append(patchtst_nan_results)

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON (with 5% NaN injection)")
    print("=" * 80)
    print(df.to_string(index=False, float_format="%.6f"))

    # Save results
    df.to_csv(f"etth1_duet_vs_patchtstnan_nan{int(NAN_RATE * 100)}pct.csv", index=False)
    print(
        f"\nResults saved to: etth1_duet_vs_patchtstnan_nan{int(NAN_RATE * 100)}pct.csv"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_acc = df.loc[df["Accuracy"].idxmax()]
    best_f1 = df.loc[df["F1 Score"].idxmax()]
    fastest = df.loc[df["Training Time (s)"].idxmin()]
    smallest = df.loc[df["Parameters"].idxmin()]

    print(f"Best Accuracy: {best_acc['Model']} ({best_acc['Accuracy']:.3f})")
    print(f"Best F1 Score: {best_f1['Model']} ({best_f1['F1 Score']:.3f})")
    print(f"Fastest Training: {fastest['Model']} ({fastest['Training Time (s)']:.1f}s)")
    print(f"Smallest Model: {smallest['Model']} ({smallest['Parameters']:,} params)")


if __name__ == "__main__":
    main()
