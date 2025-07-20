"""PatchDuET: Patch-based transformer with dual-patch NaN handling and optional column
embeddings."""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .column_embeddings import create_column_embedding


class PatchDuET(pl.LightningModule):
    """Patch-based DuET combining patching with dual-patch NaN handling.

    This model combines the best aspects of:
    - PatchTST: Patching for temporal locality (proven 90%+ accuracy)
    - DuET: Dual-patch NaN handling (value + mask channels)
    - Column embeddings: Optional semantic understanding for multi-dataset training
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        column_names: list[str] | None = None,
        use_column_embeddings: bool = False,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 3,
        lr: float = 1e-3,
        task: str = "classification",
        column_embedding_dim: int = 16,
    ):
        """Initialize PatchDuET.

        Args:
            c_in: Number of input channels
            seq_len: Sequence length
            num_classes: Number of classes for classification
            column_names: List of column names for semantic embeddings
            use_column_embeddings: Whether to use column embeddings
            patch_len: Length of each patch
            stride: Stride for patch creation
            d_model: Model dimension
            n_head: Number of attention heads
            num_layers: Number of transformer layers
            lr: Learning rate
            task: 'classification' or 'regression'
            column_embedding_dim: Dimension for column embeddings
        """
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.use_column_embeddings = use_column_embeddings
        self.column_names = column_names

        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Column embeddings (optional)
        self.column_embedder = None
        if use_column_embeddings and column_names:
            self.column_embedder = create_column_embedding(
                column_names=column_names,
                target_dim=1,  # Match value/mask dimensionality
                use_bert=False,  # Use lightweight embeddings
                embedding_dim=column_embedding_dim,
            )

        # Input channels: 2×c_in (value+mask) or 3×c_in (value+mask+column)
        input_channels_per_feature = 3 if use_column_embeddings else 2

        # Patch embedding
        self.patch_embedding = nn.Linear(
            patch_len * (input_channels_per_feature * c_in), d_model
        )

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        if task == "classification":
            self.head = nn.Linear(d_model, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.MSELoss()

    def create_patches(self, x):
        """Create patches from input tensor.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Patches tensor [B, num_patches, C * patch_len]
        """
        B, C, T = x.shape
        patches = []

        for i in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, :, i : i + self.patch_len]  # [B, C, patch_len]
            patch = patch.reshape(B, -1)  # [B, C * patch_len]
            patches.append(patch)

        patches = torch.stack(patches, dim=1)  # [B, num_patches, C * patch_len]
        return patches

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_num: Numeric features [B, C_num, T]
            x_cat: Categorical features [B, C_cat, T] (unused but kept for
                compatibility)

        Returns:
            Model output [B, num_classes] or [B, 1]
        """
        B, C_num, T = x_num.shape

        # 1. Dual-patch NaN handling (proven approach)
        m_nan = torch.isnan(x_num).float()  # [B, C_num, T]
        x_val = torch.nan_to_num(x_num, nan=0.0)  # [B, C_num, T]

        # 2. Add column embeddings if enabled
        if self.use_column_embeddings and self.column_embedder:
            # Get column embeddings [B, C_num, T]
            col_emb = self.column_embedder(B, T)

            # Triple-patch: value + mask + column
            x_processed = torch.cat([x_val, m_nan, col_emb], dim=1)  # [B, 3×C_num, T]
        else:
            # Dual-patch: value + mask
            x_processed = torch.cat([x_val, m_nan], dim=1)  # [B, 2×C_num, T]

        # 3. Create patches (winning architecture)
        patches = self.create_patches(x_processed)

        # 4. Patch embedding + positional encoding
        x = self.patch_embedding(patches)
        x = x + self.pos_embedding

        # 5. Transformer + pooling
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling

        return self.head(x)

    def training_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Training step."""
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat")
        labels = batch["y"]

        # Handle label format
        if labels.ndim == 2:
            labels = torch.argmax(labels, dim=1)
        labels = labels.long() if self.task == "classification" else labels.float()

        logits = self(x_num, x_cat)
        loss = self.loss_fn(logits.squeeze(), labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat")
        labels = batch["y"]

        # Handle label format
        if labels.ndim == 2:
            labels = torch.argmax(labels, dim=1)
        labels = labels.long() if self.task == "classification" else labels.float()

        logits = self(x_num, x_cat)
        loss = self.loss_fn(logits.squeeze(), labels)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @classmethod
    def create_baseline(
        cls,
        c_in: int,
        seq_len: int,
        num_classes: int,
        task: str = "classification",
        **kwargs,
    ):
        """Create baseline PatchDuET without column embeddings.

        Args:
            c_in: Number of input channels
            seq_len: Sequence length
            num_classes: Number of classes
            task: 'classification' or 'regression'
            **kwargs: Additional model parameters

        Returns:
            PatchDuET model without column embeddings
        """
        return cls(
            c_in=c_in,
            seq_len=seq_len,
            num_classes=num_classes,
            task=task,
            use_column_embeddings=False,
            **kwargs,
        )

    @classmethod
    def create_column_aware(
        cls,
        c_in: int,
        seq_len: int,
        num_classes: int,
        column_names: list[str],
        task: str = "classification",
        column_embedding_dim: int = 16,
        **kwargs,
    ):
        """Create column-aware PatchDuET with semantic embeddings.

        Args:
            c_in: Number of input channels
            seq_len: Sequence length
            num_classes: Number of classes
            column_names: List of column names
            task: 'classification' or 'regression'
            column_embedding_dim: Dimension for column embeddings
            **kwargs: Additional model parameters

        Returns:
            PatchDuET model with column embeddings
        """
        return cls(
            c_in=c_in,
            seq_len=seq_len,
            num_classes=num_classes,
            column_names=column_names,
            task=task,
            use_column_embeddings=True,
            column_embedding_dim=column_embedding_dim,
            **kwargs,
        )
