"""Compare DuET with PatchTST and other models on ETTh1 dataset."""

import sys
import time

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import PatchTSTModel

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, f1_score


sys.path.append(".")

from duet.data.datamodule_v2 import TimeSeriesDataModuleV2
from duet.data.etth1 import ETTh1Dataset
from duet.models.transformer import DualPatchTransformer


class HFPatchTSTClassifier(pl.LightningModule):
    """HuggingFace PatchTST adapted for classification."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        lr: float = 1e-3,
        dropout: float = 0.1,
        pretrained_model: str = "namctin/patchtst_etth1_pretrain",
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.seq_len = seq_len
        self.c_in = c_in
        self.lr = lr

        # Load pretrained PatchTST model
        if use_pretrained:
            try:
                self.patchtst = PatchTSTModel.from_pretrained(pretrained_model)
                print(f"Loaded pretrained PatchTST from {pretrained_model}")
                # Check if sequence length matches
                if self.patchtst.config.context_length != seq_len:
                    print(
                        f"Warning: Pretrained model expects sequence length {self.patchtst.config.context_length}, but got {seq_len}"
                    )
                    print("Using default PatchTST configuration instead")
                    use_pretrained = False
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                print("Using default PatchTST configuration")
                use_pretrained = False

        if not use_pretrained:
            # Create PatchTST with default configuration
            from transformers import PatchTSTConfig

            config = PatchTSTConfig(
                num_input_channels=c_in,
                context_length=seq_len,
                patch_length=16,
                patch_stride=8,
                num_hidden_layers=3,
                num_attention_heads=4,
                hidden_size=64,
                intermediate_size=256,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                prediction_length=1,
                num_targets=c_in,
                pooling_type="mean",
                norm_type="batchnorm",
                scaling=True,
                loss="mse",
                pre_norm=True,
                norm_eps=1e-5,
            )
            self.patchtst = PatchTSTModel(config)

        # Classification head
        # Get the hidden size from the model config
        hidden_size = self.patchtst.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_num, x_cat=None):
        """Forward pass.

        Args:
            x_num: Input tensor [B, C, T]
            x_cat: Categorical features (ignored)

        Returns:
            Classification logits [B, num_classes]
        """
        # PatchTST expects [B, T, C] format
        x = x_num.transpose(1, 2)  # [B, T, C]
        batch_size = x.shape[0]

        # Create dummy future values for the model
        # The pretrained model expects both past and future values
        dummy_future = torch.zeros(
            batch_size, 1, x.shape[-1], device=x.device, dtype=x.dtype
        )

        try:
            # Use the pretrained model
            outputs = self.patchtst(
                past_values=x,
                future_values=dummy_future,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get the last hidden state
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state  # [B, T, D]
            else:
                # Fallback: use the prediction head's input
                hidden_states = outputs.hidden_states[-1]  # [B, T, D]

            # Global average pooling
            pooled = hidden_states.mean(dim=1)  # [B, D]

        except Exception as e:
            print(f"Error in PatchTST forward: {e}")
            # Fallback to simple averaging and project to expected dimension
            pooled = x.mean(dim=1)  # [B, C]
            # Project to hidden_size to match classifier input expectation
            hidden_size = self.patchtst.config.hidden_size
            if pooled.shape[-1] != hidden_size:
                # Add a projection layer if dimensions don't match
                if not hasattr(self, "fallback_proj"):
                    self.fallback_proj = nn.Linear(pooled.shape[-1], hidden_size).to(
                        pooled.device
                    )
                pooled = self.fallback_proj(pooled)

        # Classification
        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        targets = batch["y"]
        # handle one-hot labels
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)
        if targets.dtype != torch.long:
            targets = targets.long()
        loss = self.loss_fn(out, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        targets = batch["y"]
        # handle one-hot labels
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)
        if targets.dtype != torch.long:
            targets = targets.long()

        loss = self.loss_fn(out, targets)
        self.log("val_loss", loss)

        # Calculate accuracy
        preds = torch.argmax(out, dim=1)
        acc = (preds == targets).float().mean()
        self.log("val_acc", acc)

        return {"val_loss": loss, "preds": preds, "labels": targets}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)


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
        targets = batch["y"]
        # handle one-hot labels
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)
        if targets.dtype != torch.long:
            targets = targets.long()
        loss = self.loss_fn(out, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        targets = batch["y"]
        # handle one-hot labels
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)
        if targets.dtype != torch.long:
            targets = targets.long()
        loss = self.loss_fn(out, targets)
        self.log("val_loss", loss)

        preds = torch.argmax(out, dim=1)
        acc = (preds == targets).float().mean()
        self.log("val_acc", acc)

        return {"val_loss": loss, "preds": preds, "labels": targets}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_model(model, dm, model_name, max_epochs=10):
    """Train a model and return results."""

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/comparison/{model_name}",
        filename="{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        mode="min",
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Logger
    logger = TensorBoardLogger(save_dir="logs", name=f"etth1_comparison/{model_name}")

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
            # Move batch to same device as model
            batch = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
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
    print("Model Comparison on ETTh1 Dataset")
    print("=" * 80)

    # Training configuration
    MAX_EPOCHS = 15
    SEQUENCE_LENGTH = 96
    NUM_CLASSES = 3

    # Load dataset
    print("\\nLoading ETTh1 dataset...")

    train_dataset = ETTh1Dataset(
        data_dir="./data/etth1",
        train=True,
        sequence_length=SEQUENCE_LENGTH,
        task="classification",
        num_classes=NUM_CLASSES,
        download=True,
    )

    val_dataset = ETTh1Dataset(
        data_dir="./data/etth1",
        train=False,
        sequence_length=SEQUENCE_LENGTH,
        task="classification",
        num_classes=NUM_CLASSES,
        download=False,
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
    feature_info = train_dataset.get_feature_info()
    c_in = feature_info["n_numeric"]
    seq_len = feature_info["sequence_length"]
    num_classes = feature_info["num_classes"]

    print("\\nDataset info:")
    print(f"- Input channels: {c_in}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of classes: {num_classes}")

    results = []

    # 1. Train DuET (our model)
    print("\\n" + "=" * 60)
    print("Training DuET (Dual-Patch Transformer)...")
    print("=" * 60)

    duet_model = DualPatchTransformer(
        C_num=c_in,
        C_cat=0,  # No categorical features in ETTh1
        cat_cardinalities=[],
        T=seq_len,
        d_model=64,
        nhead=4,
        num_layers=3,
        task="classification",
        num_classes=num_classes,
    )

    duet_results = train_model(duet_model, dm, "DuET", max_epochs=MAX_EPOCHS)
    results.append(duet_results)

    # 2. Train HuggingFace PatchTST
    print("\\n" + "=" * 60)
    print("Training HuggingFace PatchTST...")
    print("=" * 60)

    patchtst_model = HFPatchTSTClassifier(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        lr=1e-3,
        dropout=0.1,
        use_pretrained=True,
    )

    patchtst_results = train_model(
        patchtst_model, dm, "HF-PatchTST", max_epochs=MAX_EPOCHS
    )
    results.append(patchtst_results)

    # 3. Train CNN baseline
    print("\\n" + "=" * 60)
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

    cnn_results = train_model(cnn_model, dm, "CNN", max_epochs=MAX_EPOCHS)
    results.append(cnn_results)

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to CSV
    csv_filename = "etth1_model_comparison.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\\nResults saved to: {csv_filename}")

    # Create a summary
    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_acc_model = df.loc[df["Accuracy"].idxmax(), "Model"]
    best_f1_model = df.loc[df["F1 Score"].idxmax(), "Model"]
    fastest_model = df.loc[df["Training Time (s)"].idxmin(), "Model"]
    smallest_model = df.loc[df["Parameters"].idxmin(), "Model"]

    print(
        f"Best Accuracy: {best_acc_model} "
        f"({df.loc[df['Model'] == best_acc_model, 'Accuracy'].values[0]:.3f})"
    )
    print(
        f"Best F1 Score: {best_f1_model} "
        f"({df.loc[df['Model'] == best_f1_model, 'F1 Score'].values[0]:.3f})"
    )
    print(
        f"Fastest Training: {fastest_model} "
        f"({df.loc[df['Model'] == fastest_model, 'Training Time (s)'].values[0]:.1f}s)"
    )
    print(
        f"Smallest Model: {smallest_model} "
        f"({df.loc[df['Model'] == smallest_model, 'Parameters'].values[0]:,} params)"
    )

    return df


if __name__ == "__main__":
    results_df = main()
