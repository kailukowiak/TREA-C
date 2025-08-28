"""Multi-dataset PatchDuET with transferable column embeddings."""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .multi_dataset_embeddings import create_multi_dataset_embedder


class MultiDatasetPatchDuET(pl.LightningModule):
    """PatchDuET optimized for training across multiple datasets.

    Key features:
    - Pre-computed frozen BERT embeddings for semantic column understanding
    - Auto-expanding vocabulary for new datasets
    - Minimal parameter overhead for column awareness
    - Compatible with existing dual-patch NaN handling
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        column_embedding_strategy: str = "frozen_bert",
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 3,
        lr: float = 1e-3,
        task: str = "classification",
        # Column embedding specific args
        bert_model: str = "bert-base-uncased",
        column_embedding_dim: int = 16,
        initial_vocab_size: int = 1000,
        **embedder_kwargs,
    ):
        """Initialize Multi-Dataset PatchDuET.

        Args:
            c_in: Number of input channels
            seq_len: Sequence length
            num_classes: Number of classes for classification
            column_embedding_strategy: 'frozen_bert', 'auto_expanding', or 'none'
            patch_len: Length of each patch
            stride: Stride for patch creation
            d_model: Model dimension
            n_head: Number of attention heads
            num_layers: Number of transformer layers
            lr: Learning rate
            task: 'classification' or 'regression'
            bert_model: BERT model for frozen embeddings
            column_embedding_dim: Embedding dimension for auto-expanding
            initial_vocab_size: Initial vocab size for auto-expanding
            **embedder_kwargs: Additional args for embedder
        """
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.column_embedding_strategy = column_embedding_strategy
        self.c_in = c_in

        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Column embeddings
        self.use_column_embeddings = column_embedding_strategy != "none"
        self.column_embedder = None

        if self.use_column_embeddings:
            embedder_args = {"target_dim": 1, **embedder_kwargs}

            if column_embedding_strategy == "frozen_bert":
                embedder_args.update(
                    {
                        "bert_model": bert_model,
                    }
                )
            elif column_embedding_strategy == "auto_expanding":
                embedder_args.update(
                    {
                        "embedding_dim": column_embedding_dim,
                        "initial_vocab_size": initial_vocab_size,
                    }
                )

            self.column_embedder = create_multi_dataset_embedder(
                strategy=column_embedding_strategy, **embedder_args
            )

        # Input channels: 2×c_in (value+mask) or 3×c_in (value+mask+column)
        input_channels_per_feature = 3 if self.use_column_embeddings else 2

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

    def set_dataset_columns(self, column_names: list[str]) -> None:
        """Set column names for the current dataset.

        This should be called before training/evaluation on each new dataset.
        For frozen BERT, this precomputes embeddings.
        For auto-expanding, this grows the vocabulary if needed.

        Args:
            column_names: List of column names for the current dataset
        """
        if self.column_embedder is not None:
            self.column_embedder.set_columns(column_names)
            print(f"Set columns for dataset: {column_names}")
        else:
            print(f"No column embedder - ignoring column names: {column_names}")

    def create_patches(self, x):
        """Create patches from input tensor.

        Args:
            x: Input tensor [batch_size, channels, sequence_length]

        Returns:
            Patches tensor [batch_size, num_patches, patch_len * channels]
        """
        batch_size, channels, seq_len = x.shape

        # Create patches using unfold
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: [batch_size, channels, num_patches, patch_len]

        # Reshape to [batch_size, num_patches, channels * patch_len]
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch_size, self.num_patches, channels * self.patch_len)

        return patches

    def forward(self, x_num, x_cat=None):
        """Forward pass.

        Args:
            x_num: Numeric data [batch_size, c_in, sequence_length]
            x_cat: Categorical data (unused in current implementation)

        Returns:
            Model output
        """
        B, C, T = x_num.shape

        # 1. Dual-patch NaN handling
        m_nan = torch.isnan(x_num).float()  # [B, C, T]
        x_val = torch.nan_to_num(x_num, nan=0.0)  # [B, C, T]

        # 2. Triple-patch: value + mask + column (if enabled)
        if self.use_column_embeddings and self.column_embedder is not None:
            col_emb = self.column_embedder(B, T)  # [B, C, T]
            x_processed = torch.cat([x_val, m_nan, col_emb], dim=1)  # [B, 3*C, T]
        else:
            x_processed = torch.cat([x_val, m_nan], dim=1)  # [B, 2*C, T]

        # 3. Create patches
        patches = self.create_patches(
            x_processed
        )  # [B, num_patches, patch_len * (2 or 3)*C]

        # 4. Patch embedding
        patch_embeddings = self.patch_embedding(patches)  # [B, num_patches, d_model]

        # 5. Add positional encoding
        patch_embeddings = patch_embeddings + self.pos_embedding

        # 6. Transformer
        transformer_out = self.transformer(
            patch_embeddings
        )  # [B, num_patches, d_model]

        # 7. Global average pooling
        pooled = transformer_out.mean(dim=1)  # [B, d_model]

        # 8. Classification/regression head
        output = self.head(pooled)

        return output

    def training_step(self, batch, batch_idx):
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat", None)
        y = batch["y"]

        y_hat = self(x_num, x_cat)

        if self.task == "classification":
            if y.ndim == 2:  # One-hot encoded
                y = torch.argmax(y, dim=1)
            y = y.long()
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat.squeeze(), y.float())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat", None)
        y = batch["y"]

        y_hat = self(x_num, x_cat)

        if self.task == "classification":
            if y.ndim == 2:  # One-hot encoded
                y = torch.argmax(y, dim=1)
            y = y.long()
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat.squeeze(), y.float())

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_embedding_stats(self):
        """Get statistics about the column embedder."""
        if self.column_embedder is None:
            return {"strategy": "none"}

        stats = {"strategy": self.column_embedding_strategy}

        if hasattr(self.column_embedder, "get_cache_stats"):
            stats.update(self.column_embedder.get_cache_stats())

        if hasattr(self.column_embedder, "embedding"):
            stats["vocab_size"] = self.column_embedder.embedding.num_embeddings
            stats["next_idx"] = self.column_embedder.next_idx

        return stats

    @classmethod
    def create_for_dataset(
        cls,
        c_in: int,
        seq_len: int,
        num_classes: int,
        column_names: list[str],
        strategy: str = "frozen_bert",
        **kwargs,
    ) -> "MultiDatasetPatchDuET":
        """Factory method to create model for a specific dataset.

        Args:
            c_in: Number of input channels
            seq_len: Sequence length
            num_classes: Number of classes
            column_names: Column names for this dataset
            strategy: Column embedding strategy
            **kwargs: Additional model arguments

        Returns:
            Configured MultiDatasetPatchDuET instance
        """
        model = cls(
            c_in=c_in,
            seq_len=seq_len,
            num_classes=num_classes,
            column_embedding_strategy=strategy,
            **kwargs,
        )

        # Set columns for this dataset
        model.set_dataset_columns(column_names)

        return model

    @classmethod
    def create_baseline(
        cls, c_in: int, seq_len: int, num_classes: int, **kwargs
    ) -> "MultiDatasetPatchDuET":
        """Create baseline model without column embeddings."""
        return cls(
            c_in=c_in,
            seq_len=seq_len,
            num_classes=num_classes,
            column_embedding_strategy="none",
            **kwargs,
        )


if __name__ == "__main__":
    # Test the multi-dataset model
    print("Testing MultiDatasetPatchDuET...")

    # Create models with different strategies
    strategies = ["none", "frozen_bert", "auto_expanding"]

    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")

        if strategy == "none":
            model = MultiDatasetPatchDuET.create_baseline(
                c_in=7, seq_len=96, num_classes=3
            )
        else:
            model = MultiDatasetPatchDuET.create_for_dataset(
                c_in=7,
                seq_len=96,
                num_classes=3,
                column_names=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
                strategy=strategy,
            )

        # Test forward pass
        x_num = torch.randn(4, 7, 96)
        # Inject some NaNs
        x_num[0, :2, :10] = float("nan")

        output = model(x_num)
        print(f"Output shape: {output.shape}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Embedding stats: {model.get_embedding_stats()}")

        # Test with new columns (for expandable strategies)
        if strategy in ["frozen_bert", "auto_expanding"]:
            print("Testing with new columns...")
            model.set_dataset_columns(["temperature", "humidity", "pressure"])
            output2 = model(torch.randn(4, 3, 96))
            print(f"New output shape: {output2.shape}")
            print(f"Updated embedding stats: {model.get_embedding_stats()}")
