"""Compare DuET with PatchTST and other models on ETTh1 dataset."""

import sys
import time

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, f1_score
from transformers import PatchTSTConfig, PatchTSTModel


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
        use_pretrained: bool = False,  # Seq classification not supported
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_classes = num_classes

        # Create PatchTST config for forecasting model with much larger capacity
        config = PatchTSTConfig(
            num_input_channels=c_in,
            context_length=seq_len,
            patch_length=16,
            patch_stride=8,
            num_hidden_layers=8,  # Increased from 6
            num_attention_heads=16,  # Increased from 8
            hidden_size=256,  # Increased from 128
            intermediate_size=1024,  # Increased from 512
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        # Use the base PatchTST model
        self.patchtst = PatchTSTModel(config)

        # Add classification head with attention pooling
        self.attention_pool = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size // 2, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_num, x_cat=None):
        # PatchTST expects [B, T, C]
        x = x_num.transpose(1, 2)

        # Get patch embeddings from PatchTST backbone
        outputs = self.patchtst(
            past_values=x,
            return_dict=True,
        )

        # Use the last hidden state for classification
        # outputs.last_hidden_state shape: [B, num_features, num_patches, hidden_size]
        batch_size = outputs.last_hidden_state.shape[0]
        hidden_states = outputs.last_hidden_state.view(
            batch_size, -1, outputs.last_hidden_state.shape[-1]
        )

        # Attention-based pooling
        attention_weights = torch.softmax(self.attention_pool(hidden_states), dim=1)
        pooled = torch.sum(attention_weights * hidden_states, dim=1)

        # Pass through classification head
        logits = self.classifier(pooled)

        return logits

    def training_step(self, batch, _batch_idx):
        x = batch["x_num"]
        labels = batch["y"]

        # Ensure labels are in the right format for CrossEntropyLoss
        if labels.ndim == 2:
            # If labels are one-hot encoded, convert to class indices
            labels = torch.argmax(labels, dim=1)
        labels = labels.long()

        # Forward pass
        logits = self.forward(x)

        # CrossEntropyLoss expects: logits [B, C] and labels [B]
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        loss = self.loss_fn(logits, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x = batch["x_num"]
        labels = batch["y"]

        # Ensure labels are in the right format for CrossEntropyLoss
        if labels.ndim == 2:
            # If labels are one-hot encoded, convert to class indices
            labels = torch.argmax(labels, dim=1)
        labels = labels.long()

        # Forward pass
        logits = self.forward(x)

        # CrossEntropyLoss expects: logits [B, C] and labels [B]
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        loss = self.loss_fn(logits, labels)

        self.log("val_loss", loss)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc)
        return {"val_loss": loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)


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
        # For simplicity, only use numeric features
        patches = self.create_patches(x_num)

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
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc)
        return {"val_loss": loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.01
        )


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

    def training_step(self, batch, _batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        targets = batch["y"]
        # ensure 1D class indices
        if targets.ndim == 2:
            targets = torch.argmax(targets, dim=1)
        targets = targets.reshape(-1).long()
        loss = self.loss_fn(out, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        targets = batch["y"]
        # ensure 1D class indices
        if targets.ndim == 2:
            targets = torch.argmax(targets, dim=1)
        targets = targets.reshape(-1).long()
        loss = self.loss_fn(out, targets)
        self.log("val_loss", loss)

        preds = torch.argmax(out, dim=1)
        acc = (preds == targets).float().mean()
        self.log("val_acc", acc)

        return {"val_loss": loss, "preds": preds, "labels": targets}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


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

        # Patch embedding - NOTE: input size is 2*c_in due to dual-patch (value + mask)
        self.patch_embedding = nn.Linear(patch_len * (2 * c_in), d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
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
        d_model=128,  # Increased from 64 to match PatchTST
        nhead=8,  # Increased from 4 to match PatchTST
        num_layers=3,
        task="classification",
        num_classes=num_classes,
        lr=1e-3,
    )

    duet_results = train_model(duet_model, dm, "DuET", max_epochs=MAX_EPOCHS)
    results.append(duet_results)

    # 2. Train HuggingFace PatchTST
    print("\\n" + "=" * 60)
    print("Training HuggingFace PatchTST...")
    print("=" * 60)

    hf_patchtst_model = HFPatchTSTClassifier(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        lr=5e-4,  # Reduce learning rate
        dropout=0.1,
        use_pretrained=False,
    )

    hf_patchtst_results = train_model(
        hf_patchtst_model, dm, "HF-PatchTST", max_epochs=MAX_EPOCHS
    )
    results.append(hf_patchtst_results)

    # 3. Train Custom PatchTST
    print("\\n" + "=" * 60)
    print("Training Custom PatchTST...")
    print("=" * 60)

    custom_patchtst_model = PatchTSTClassifier(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        patch_len=16,
        stride=8,
        d_model=128,
        nhead=8,
        num_layers=3,
        lr=1e-3,
    )

    custom_patchtst_results = train_model(
        custom_patchtst_model, dm, "Custom-PatchTST", max_epochs=MAX_EPOCHS
    )
    results.append(custom_patchtst_results)

    # 4. Train CNN baseline
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

    # 5. Train PatchTSTNan (Custom PatchTST with dual-patch NaN handling)
    print("\\n" + "=" * 60)
    print("Training PatchTSTNan...")
    print("=" * 60)

    patchtst_nan_model = PatchTSTNan(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        patch_len=16,
        stride=8,
        d_model=128,
        nhead=8,
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
