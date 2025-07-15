"""Compare DuET with PatchTST and other models on NASA Turbofan dataset."""

import sys


sys.path.append(".")

import time

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, f1_score

from duet.data.datamodule_v2 import TimeSeriesDataModuleV2
from duet.data.nasa_turbofan import NASATurbofanDataset
from duet.models.transformer import DualPatchTransformer


class PatchTSTClassifier(pl.LightningModule):
    """PatchTST adapted for classification."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * c_in, d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Linear(d_model * self.num_patches, num_classes)

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
        # For simplicity, only use numeric features
        patches = self.create_patches(x_num)

        # Embed patches
        x = self.patch_embedding(patches)
        x = x + self.pos_embedding

        # Transformer
        x = self.transformer(x)

        # Global pooling and classification
        x = self.flatten(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(out, batch["y"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(out, batch["y"])
        self.log("val_loss", loss)

        # Calculate accuracy
        preds = torch.argmax(out, dim=1)
        acc = (preds == batch["y"]).float().mean()
        self.log("val_acc", acc)

        return {"val_loss": loss, "preds": preds, "labels": batch["y"]}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class CNNClassifier(pl.LightningModule):
    """1D CNN baseline for time series classification."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        hidden_dim: int = 64,
        kernel_size: int = 7,
        num_layers: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_channels = c_in

        for i in range(num_layers):
            out_channels = hidden_dim * (2**i)
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size, padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)

        # Calculate output size
        with torch.no_grad():
            dummy = torch.zeros(1, c_in, seq_len)
            out = self.feature_extractor(dummy)
            flat_size = out.shape[1] * out.shape[2]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_num, x_cat=None):
        x = self.feature_extractor(x_num)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(out, batch["y"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(out, batch["y"])
        self.log("val_loss", loss)

        preds = torch.argmax(out, dim=1)
        acc = (preds == batch["y"]).float().mean()
        self.log("val_acc", acc)

        return {"val_loss": loss, "preds": preds, "labels": batch["y"]}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_model(model, dm, model_name, max_epochs=30):
    """Train a model and return results."""

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/comparison/{model_name}",
        filename="{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        mode="min",
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs", name=f"turbofan_comparison/{model_name}"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Time training
    start_time = time.time()
    trainer.fit(model, dm)
    training_time = time.time() - start_time

    # Load best model
    best_model_path = checkpoint.best_model_path
    model = model.__class__.load_from_checkpoint(best_model_path)

    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dm.val_dataloader():
            out = model(batch["x_num"], batch.get("x_cat"))
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["y"].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # Get validation loss from trainer
    val_loss = trainer.callback_metrics.get(
        "val_loss", torch.tensor(float("nan"))
    ).item()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Val Loss": val_loss,
        "Training Time (s)": training_time,
        "Parameters": num_params,
        "Epochs": trainer.current_epoch + 1,
    }


def main():
    print("=" * 80)
    print("Model Comparison on NASA Turbofan Dataset")
    print("=" * 80)

    # Load dataset
    print("\nLoading NASA Turbofan dataset...")
    try:
        train_dataset = NASATurbofanDataset(
            data_dir="./data/nasa_cmapss",
            subset="FD001",
            train=True,
            sequence_length=50,
            task="classification",
            num_classes=3,
            download=True,
        )

        val_dataset = NASATurbofanDataset(
            data_dir="./data/nasa_cmapss",
            subset="FD001",
            train=False,
            sequence_length=50,
            task="classification",
            num_classes=3,
            download=False,
        )

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic fallback...")

        from duet.data.dataset import SyntheticTimeSeriesDataset

        train_dataset = SyntheticTimeSeriesDataset(
            num_samples=5000,
            T=50,
            C_num=21,
            C_cat=2,
            cat_cardinalities=[5, 5],
            num_classes=3,
            task="classification",
            missing_ratio=0.05,
        )
        val_dataset = SyntheticTimeSeriesDataset(
            num_samples=1000,
            T=50,
            C_num=21,
            C_cat=2,
            cat_cardinalities=[5, 5],
            num_classes=3,
            task="classification",
            missing_ratio=0.05,
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data module
    dm = TimeSeriesDataModuleV2(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64,
        num_workers=4,
    )

    # Get data dimensions
    sample = train_dataset[0]
    c_in = sample["x_num"].shape[0]
    seq_len = sample["x_num"].shape[1]
    num_classes = 3

    print("\nDataset info:")
    print(f"- Input channels: {c_in}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of classes: {num_classes}")

    results = []

    # 1. Train DuET (our model)
    print("\n" + "=" * 60)
    print("Training DuET (Dual-Patch Transformer)...")
    print("=" * 60)

    duet_model = DualPatchTransformer(
        C_num=c_in,
        C_cat=2,
        cat_cardinalities=[5, 5],
        T=seq_len,
        d_model=64,
        nhead=4,
        num_layers=3,
        task="classification",
        num_classes=num_classes,
    )

    duet_results = train_model(duet_model, dm, "DuET", max_epochs=30)
    results.append(duet_results)

    # 2. Train PatchTST
    print("\n" + "=" * 60)
    print("Training PatchTST...")
    print("=" * 60)

    patchtst_model = PatchTSTClassifier(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        patch_len=10,
        stride=5,
        d_model=64,
        nhead=4,
        num_layers=3,
    )

    patchtst_results = train_model(patchtst_model, dm, "PatchTST", max_epochs=30)
    results.append(patchtst_results)

    # 3. Train CNN baseline
    print("\n" + "=" * 60)
    print("Training 1D CNN Baseline...")
    print("=" * 60)

    cnn_model = CNNClassifier(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        hidden_dim=64,
        kernel_size=7,
        num_layers=3,
    )

    cnn_results = train_model(cnn_model, dm, "CNN", max_epochs=30)
    results.append(cnn_results)

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to CSV
    csv_filename = "turbofan_model_comparison.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")

    # Create a summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_acc_model = df.loc[df["Accuracy"].idxmax(), "Model"]
    best_f1_model = df.loc[df["F1 Score"].idxmax(), "Model"]
    fastest_model = df.loc[df["Training Time (s)"].idxmin(), "Model"]
    smallest_model = df.loc[df["Parameters"].idxmin(), "Model"]

    print(
        f"Best Accuracy: {best_acc_model} ({df.loc[df['Model'] == best_acc_model, 'Accuracy'].values[0]:.3f})"
    )
    print(
        f"Best F1 Score: {best_f1_model} ({df.loc[df['Model'] == best_f1_model, 'F1 Score'].values[0]:.3f})"
    )
    print(
        f"Fastest Training: {fastest_model} ({df.loc[df['Model'] == fastest_model, 'Training Time (s)'].values[0]:.1f}s)"
    )
    print(
        f"Smallest Model: {smallest_model} ({df.loc[df['Model'] == smallest_model, 'Parameters'].values[0]:,} params)"
    )

    return df


if __name__ == "__main__":
    results_df = main()
