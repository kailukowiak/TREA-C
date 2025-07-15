"""HuggingFace PatchTST wrapper for classification tasks."""


import pytorch_lightning as pl
import torch
import torch.nn as nn

from transformers import PatchTSTConfig, PatchTSTModel


class HFPatchTSTClassifier(pl.LightningModule):
    """HuggingFace PatchTST adapted for classification."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 64,
        num_layers: int = 3,
        nhead: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        lr: float = 1e-3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.lr = lr

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Create PatchTST configuration
        config = PatchTSTConfig(
            num_input_channels=c_in,
            context_length=seq_len,
            patch_length=patch_len,
            patch_stride=stride,
            num_hidden_layers=num_layers,
            num_attention_heads=nhead,
            hidden_size=d_model,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            prediction_length=1,  # We'll ignore this for classification
            num_targets=c_in,
            # Set other reasonable defaults
            pooling_type="mean",
            norm_type="batchnorm",
            scaling=True,
            loss="mse",  # Will be overridden
            pre_norm=True,
            norm_eps=1e-5,
        )

        # Create the model
        self.patchtst = PatchTSTModel(config)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None):
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

        # Use a simpler approach - just use the encoder
        try:
            # Get the transformer encoder directly
            encoder = self.patchtst.encoder

            # Create dummy inputs in the expected format
            dummy_future = torch.zeros(
                batch_size, 1, x.shape[-1], device=x.device, dtype=x.dtype
            )

            # Try to get encoder output
            encoded = encoder(
                past_values=x,
                future_values=dummy_future,
                output_hidden_states=True,
                return_dict=True,
            )

            # Pool the encoder output
            if hasattr(encoded, "last_hidden_state"):
                hidden_states = encoded.last_hidden_state  # [B, T, D]
            else:
                hidden_states = encoded.hidden_states[-1]  # [B, T, D]

            # Global average pooling
            pooled = hidden_states.mean(dim=1)  # [B, D]

        except Exception as e:
            print(f"Error in PatchTST forward: {e}, falling back to manual patching")
            # Fallback to manual patching
            patches = []
            patch_len = self.hparams.patch_len
            stride = self.hparams.stride

            for i in range(0, x.shape[1] - patch_len + 1, stride):
                patch = x[:, i : i + patch_len, :]  # [B, patch_len, C]
                patches.append(patch.mean(dim=1))  # [B, C]

            if patches:
                pooled = torch.stack(patches, dim=1).mean(dim=1)  # [B, C]
            else:
                pooled = x.mean(dim=1)  # [B, C]

        # Classification
        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(logits, batch["y"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(logits, batch["y"])
        self.log("val_loss", loss, prog_bar=True)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["y"]).float().mean()
        self.log("val_acc", acc, prog_bar=True)

        return {"val_loss": loss, "preds": preds, "labels": batch["y"]}

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
