"""Variable feature count PatchDuET for true multi-dataset training."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .multi_dataset_embeddings import create_multi_dataset_embedder


class VariableFeaturePatchDuET(pl.LightningModule):
    """PatchDuET that handles different numbers of features across datasets.

    Key innovations:
    - Unified feature space with feature masking
    - Dynamic feature mapping per dataset
    - Maintains patching benefits across variable schemas
    - Compatible with column embeddings
    """

    def __init__(
        self,
        max_numeric_features: int,
        max_categorical_features: int,
        num_classes: int,
        column_embedding_strategy: str = "auto_expanding",
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 3,
        lr: float = 1e-3,
        task: str = "classification",
        # Feature handling
        feature_padding_value: float = 0.0,
        use_feature_masks: bool = True,
        # Multi-dataset embedding args
        bert_model: str = "bert-base-uncased",
        column_embedding_dim: int = 16,
        initial_vocab_size: int = 1000,
        max_sequence_length: int = 1024,
        **embedder_kwargs,
    ):
        """Initialize Variable Feature PatchDuET.

        Args:
            max_numeric_features: Maximum number of numeric features across all datasets
            max_categorical_features: Maximum number of categorical features
            num_classes: Number of output classes
            column_embedding_strategy: Column embedding strategy
            patch_len: Patch length
            stride: Patch stride
            d_model: Model dimension
            n_head: Number of attention heads
            num_layers: Number of transformer layers
            lr: Learning rate
            task: 'classification' or 'regression'
            feature_padding_value: Value to use for missing features
            use_feature_masks: Whether to use explicit feature presence masks
            max_sequence_length: Maximum sequence length for positional encoding
            **embedder_kwargs: Additional embedder arguments
        """
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.column_embedding_strategy = column_embedding_strategy
        self.max_numeric_features = max_numeric_features
        self.max_categorical_features = max_categorical_features
        self.feature_padding_value = feature_padding_value
        self.use_feature_masks = use_feature_masks

        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        self.max_sequence_length = max_sequence_length

        # Current dataset configuration
        self.current_dataset_info = {
            "numeric_features": max_numeric_features,
            "categorical_features": max_categorical_features,
            "column_names": [],
            "feature_mapping": None,
        }

        # Column embeddings (for the maximum feature space)
        self.use_column_embeddings = column_embedding_strategy != "none"
        self.column_embedder = None

        if self.use_column_embeddings:
            embedder_args = {"target_dim": 1, **embedder_kwargs}

            if column_embedding_strategy == "frozen_bert":
                embedder_args.update({"bert_model": bert_model})
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

        # Total features in unified space
        total_features = max_numeric_features + max_categorical_features

        # Input channels: [value, nan_mask, feature_mask?, column_emb?]
        base_channels = 2  # value + nan_mask
        if use_feature_masks:
            base_channels += 1  # + feature_mask
        if self.use_column_embeddings:
            base_channels += 1  # + column_emb

        input_channels_per_feature = base_channels

        # Patch embedding
        self.patch_embedding = nn.Linear(
            patch_len * (input_channels_per_feature * total_features), d_model
        )

        # Dynamic positional encoding
        max_patches = (max_sequence_length - patch_len) // stride + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model))

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

    def set_dataset_schema(
        self,
        numeric_features: int,
        categorical_features: int = 0,
        column_names: list[str] | None = None,
    ) -> None:
        """Set the schema for the current dataset.

        Args:
            numeric_features: Number of numeric features in this dataset
            categorical_features: Number of categorical features
            column_names: Optional list of column names
        """
        if numeric_features > self.max_numeric_features:
            raise ValueError(
                f"Dataset has {numeric_features} numeric features, "
                f"but model supports max {self.max_numeric_features}"
            )

        if categorical_features > self.max_categorical_features:
            raise ValueError(
                f"Dataset has {categorical_features} categorical features, "
                f"but model supports max {self.max_categorical_features}"
            )

        # Update current dataset info
        self.current_dataset_info = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "column_names": column_names
            or [f"feature_{i}" for i in range(numeric_features + categorical_features)],
            "feature_mapping": self._create_feature_mapping(
                numeric_features, categorical_features
            ),
        }

        # Set column embeddings if enabled
        if self.use_column_embeddings and self.column_embedder is not None:
            # Create padded column names for full feature space
            full_column_names = self._get_padded_column_names()
            self.column_embedder.set_columns(full_column_names)

        print(
            f"Set dataset schema: {numeric_features} numeric + "
            f"{categorical_features} categorical features"
        )
        if column_names:
            print(f"Column names: {column_names}")

    def _create_feature_mapping(
        self, numeric_features: int, categorical_features: int
    ) -> dict[str, Any]:
        """Create mapping for current dataset features to unified space."""
        return {
            "numeric_start": 0,
            "numeric_end": numeric_features,
            "categorical_start": self.max_numeric_features,
            "categorical_end": self.max_numeric_features + categorical_features,
            "total_active_features": numeric_features + categorical_features,
        }

    def _get_padded_column_names(self) -> list[str]:
        """Get column names padded to full feature space."""
        column_names = self.current_dataset_info["column_names"]
        mapping = self.current_dataset_info["feature_mapping"]

        # Create full column name list
        full_names = ["<UNUSED>"] * (
            self.max_numeric_features + self.max_categorical_features
        )

        # Fill in actual column names
        actual_names = column_names[: mapping["total_active_features"]]

        # Place numeric features
        for i, name in enumerate(
            actual_names[: self.current_dataset_info["numeric_features"]]
        ):
            full_names[i] = name

        # Place categorical features
        for i, name in enumerate(
            actual_names[self.current_dataset_info["numeric_features"] :]
        ):
            full_names[self.max_numeric_features + i] = name

        return full_names

    def pad_features_to_unified_space(
        self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad input features to unified feature space.

        Args:
            x_num: Numeric features [B, C_num, T]
            x_cat: Categorical features [B, C_cat, T] (optional)

        Returns:
            Tuple of (padded_features, feature_mask)
            - padded_features: [B, max_total_features, T]
            - feature_mask: [B, max_total_features, T] (1=real feature, 0=padding)
        """
        B, _, T = x_num.shape
        mapping = self.current_dataset_info["feature_mapping"]

        total_max_features = self.max_numeric_features + self.max_categorical_features

        # Create unified feature tensor
        padded_features = torch.full(
            (B, total_max_features, T),
            self.feature_padding_value,
            dtype=x_num.dtype,
            device=x_num.device,
        )

        # Create feature mask
        feature_mask = torch.zeros(
            (B, total_max_features, T), dtype=torch.float, device=x_num.device
        )

        # Place numeric features
        num_numeric = mapping["numeric_end"] - mapping["numeric_start"]
        if num_numeric > 0:
            padded_features[:, mapping["numeric_start"] : mapping["numeric_end"], :] = (
                x_num
            )
            feature_mask[:, mapping["numeric_start"] : mapping["numeric_end"], :] = 1.0

        # Place categorical features
        if x_cat is not None:
            num_categorical = mapping["categorical_end"] - mapping["categorical_start"]
            if num_categorical > 0:
                padded_features[
                    :, mapping["categorical_start"] : mapping["categorical_end"], :
                ] = x_cat
                feature_mask[
                    :, mapping["categorical_start"] : mapping["categorical_end"], :
                ] = 1.0

        return padded_features, feature_mask

    def create_patches(self, x):
        """Create patches from input tensor with dynamic sequence length."""
        batch_size, channels, seq_len = x.shape

        # Calculate number of patches dynamically
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        if num_patches <= 0:
            raise ValueError(
                f"Sequence length {seq_len} too short for patch_len={self.patch_len}"
            )

        # Create patches using unfold
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: [batch_size, channels, num_patches, patch_len]

        # Reshape to [batch_size, num_patches, channels * patch_len]
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch_size, num_patches, channels * self.patch_len)

        return patches

    def get_positional_encoding(self, num_patches: int):
        """Get positional encoding for the given number of patches."""
        if num_patches > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Sequence too long: {num_patches} patches exceeds maximum "
                f"{self.pos_embedding.shape[1]}. Increase max_sequence_length."
            )

        return self.pos_embedding[:, :num_patches, :]

    def forward(self, x_num, x_cat=None):
        """Forward pass with variable feature support.

        Args:
            x_num: Numeric data [batch_size, actual_c_num, sequence_length]
            x_cat: Categorical data [batch_size, actual_c_cat, sequence_length]
                   (optional)

        Returns:
            Model output
        """
        B, _, T = x_num.shape

        # 1. Pad features to unified space
        x_padded, feature_mask = self.pad_features_to_unified_space(x_num, x_cat)
        # x_padded: [B, max_total_features, T]
        # feature_mask: [B, max_total_features, T]

        # 2. Dual-patch NaN handling (on padded space)
        m_nan = torch.isnan(x_padded).float()  # [B, max_total_features, T]
        x_val = torch.nan_to_num(x_padded, nan=0.0)  # [B, max_total_features, T]

        # 3. Combine channels
        channels_to_cat = [x_val, m_nan]

        if self.use_feature_masks:
            channels_to_cat.append(feature_mask)

        if self.use_column_embeddings and self.column_embedder is not None:
            col_emb = self.column_embedder(B, T)  # [B, max_total_features, T]
            channels_to_cat.append(col_emb)

        x_processed = torch.cat(
            channels_to_cat, dim=1
        )  # [B, channels * max_total_features, T]

        # 4. Create patches
        patches = self.create_patches(x_processed)
        num_patches = patches.shape[1]

        # 5. Patch embedding
        patch_embeddings = self.patch_embedding(patches)  # [B, num_patches, d_model]

        # 6. Add positional encoding
        pos_enc = self.get_positional_encoding(num_patches)
        patch_embeddings = patch_embeddings + pos_enc

        # 7. Transformer
        transformer_out = self.transformer(
            patch_embeddings
        )  # [B, num_patches, d_model]

        # 8. Global average pooling
        pooled = transformer_out.mean(dim=1)  # [B, d_model]

        # 9. Classification/regression head
        output = self.head(pooled)

        return output

    def training_step(self, batch, batch_idx):
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat", None)
        y = batch["y"]

        y_hat = self(x_num, x_cat)

        if self.task == "classification":
            if y.ndim == 2:
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
            if y.ndim == 2:
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
    def create_for_multi_dataset(
        cls,
        dataset_schemas: dict[str, dict[str, Any]],
        num_classes: int,
        strategy: str = "auto_expanding",
        **kwargs,
    ) -> "VariableFeaturePatchDuET":
        """Factory method for multi-dataset training.

        Args:
            dataset_schemas: Dict mapping dataset names to schema info
                Example: {
                    "dataset1": {"numeric": 10, "categorical": 3, 
                                 "columns": ["a", "b", ...]},
                    "dataset2": {"numeric": 16, "categorical": 1, 
                                 "columns": ["x", "y", ...]}
                }
            num_classes: Number of output classes
            strategy: Column embedding strategy
            **kwargs: Additional model arguments

        Returns:
            Configured VariableFeaturePatchDuET instance
        """
        # Find maximum feature counts across all datasets
        max_numeric = max(schema["numeric"] for schema in dataset_schemas.values())
        max_categorical = max(
            schema.get("categorical", 0) for schema in dataset_schemas.values()
        )

        print("Multi-dataset model configuration:")
        print(f"  Max numeric features: {max_numeric}")
        print(f"  Max categorical features: {max_categorical}")
        print("  Dataset schemas:")

        for name, schema in dataset_schemas.items():
            print(
                f"    {name}: {schema['numeric']} numeric + "
                f"{schema.get('categorical', 0)} categorical"
            )

        model = cls(
            max_numeric_features=max_numeric,
            max_categorical_features=max_categorical,
            num_classes=num_classes,
            column_embedding_strategy=strategy,
            **kwargs,
        )

        return model

    @classmethod
    def create_baseline(
        cls,
        max_numeric_features: int,
        max_categorical_features: int,
        num_classes: int,
        **kwargs,
    ) -> "VariableFeaturePatchDuET":
        """Create baseline model without column embeddings."""
        return cls(
            max_numeric_features=max_numeric_features,
            max_categorical_features=max_categorical_features,
            num_classes=num_classes,
            column_embedding_strategy="none",
            **kwargs,
        )


if __name__ == "__main__":
    # Test variable feature counts
    print("Testing VariableFeaturePatchDuET...")

    # Define multiple dataset schemas
    dataset_schemas = {
        "sensor_data": {
            "numeric": 8,
            "categorical": 2,
            "columns": [
                "temp1",
                "temp2",
                "humidity",
                "pressure",
                "wind",
                "rain",
                "light",
                "sound",
                "zone",
                "status",
            ],
        },
        "user_metrics": {
            "numeric": 12,
            "categorical": 1,
            "columns": [
                "clicks",
                "views",
                "time",
                "bounce",
                "conversion",
                "revenue",
                "sessions",
                "pages",
                "downloads",
                "shares",
                "likes",
                "comments",
                "segment",
            ],
        },
        "financial": {
            "numeric": 6,
            "categorical": 0,
            "columns": ["price", "volume", "high", "low", "close", "volatility"],
        },
    }

    # Create model for all datasets
    model = VariableFeaturePatchDuET.create_for_multi_dataset(
        dataset_schemas=dataset_schemas, num_classes=3, strategy="auto_expanding"
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test each dataset schema
    for dataset_name, schema in dataset_schemas.items():
        print(f"\n--- Testing {dataset_name} ---")

        # Set schema for this dataset
        model.set_dataset_schema(
            numeric_features=schema["numeric"],
            categorical_features=schema.get("categorical", 0),
            column_names=schema["columns"],
        )

        # Create sample data matching this schema
        B, T = 4, 96
        x_num = torch.randn(B, schema["numeric"], T)
        x_cat = None
        if schema.get("categorical", 0) > 0:
            x_cat = torch.randint(0, 5, (B, schema["categorical"], T)).float()

        # Add some NaNs
        x_num[0, :2, :10] = float("nan")

        # Forward pass
        output = model(x_num, x_cat)
        print(
            f"  Input: numeric {x_num.shape}, "
            f"categorical {x_cat.shape if x_cat is not None else None}"
        )
        print(f"  Output: {output.shape}")
        print("  Success: ✅")
