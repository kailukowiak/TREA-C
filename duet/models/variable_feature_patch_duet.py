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
        # Categorical embedding args
        categorical_cardinalities: list[int] | None = None,
        categorical_embedding_dim: int = 16,
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
            categorical_cardinalities: List of cardinality for each categorical feature
            categorical_embedding_dim: Embedding dimension for categorical features
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
        self.categorical_embedding_dim = categorical_embedding_dim

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

        # Categorical embeddings
        self.categorical_embeddings = None
        if max_categorical_features > 0:
            if categorical_cardinalities is None:
                # Default cardinality of 100 for each categorical feature
                categorical_cardinalities = [100] * max_categorical_features
            elif len(categorical_cardinalities) != max_categorical_features:
                raise ValueError(
                    f"categorical_cardinalities length "
                    f"({len(categorical_cardinalities)}) must match "
                    f"max_categorical_features ({max_categorical_features})"
                )

            self.categorical_embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality, categorical_embedding_dim)
                    for cardinality in categorical_cardinalities
                ]
            )

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

        # Calculate input channels for unified space
        # Numeric features: [value, nan_mask, feature_mask?, column_emb?]
        numeric_channels = 2  # value + nan_mask
        if use_feature_masks:
            numeric_channels += 1  # + feature_mask
        if self.use_column_embeddings:
            numeric_channels += 1  # + column_emb

        # Categorical features: [embedding_dims, feature_mask?, column_emb?]
        categorical_channels = categorical_embedding_dim  # embedded values
        if use_feature_masks:
            categorical_channels += 1  # + feature_mask
        if self.use_column_embeddings:
            categorical_channels += 1  # + column_emb

        # Total input channels (mixed numeric and categorical)
        total_input_channels = (
            max_numeric_features * numeric_channels
            + max_categorical_features * categorical_channels
        )

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * total_input_channels, d_model)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad features to unified space with proper categorical embeddings.

        Args:
            x_num: Numeric features [B, C_num, T]
            x_cat: Categorical features [B, C_cat, T] (optional)

        Returns:
            Tuple of (x_num_padded, x_cat_embedded, feature_mask)
            - x_num_padded: [B, max_numeric_features, T] (with padding)
            - x_cat_embedded: [B, max_categorical_features,
                               categorical_embedding_dim, T] (embedded)
            - feature_mask: [B, max_numeric_features +
                             max_categorical_features, T] (1=real, 0=pad)
        """
        B, _, T = x_num.shape
        mapping = self.current_dataset_info["feature_mapping"]

        # Pad numeric features
        x_num_padded = torch.full(
            (B, self.max_numeric_features, T),
            self.feature_padding_value,
            dtype=x_num.dtype,
            device=x_num.device,
        )

        # Create feature mask for both numeric and categorical
        total_max_features = self.max_numeric_features + self.max_categorical_features
        feature_mask = torch.zeros(
            (B, total_max_features, T), dtype=torch.float, device=x_num.device
        )

        # Place numeric features
        num_numeric = mapping["numeric_end"] - mapping["numeric_start"]
        if num_numeric > 0:
            x_num_padded[:, :num_numeric, :] = x_num
            feature_mask[:, :num_numeric, :] = 1.0

        # Handle categorical features with embeddings
        x_cat_embedded = torch.zeros(
            (B, self.max_categorical_features, self.categorical_embedding_dim, T),
            dtype=torch.float,
            device=x_num.device,
        )

        if x_cat is not None and self.categorical_embeddings is not None:
            num_categorical = mapping["categorical_end"] - mapping["categorical_start"]
            if num_categorical > 0:
                for i in range(num_categorical):
                    # Embed each categorical feature separately
                    cat_indices = x_cat[:, i, :].long()  # [B, T]
                    embedded = self.categorical_embeddings[i](
                        cat_indices
                    )  # [B, T, embedding_dim]
                    x_cat_embedded[:, i, :, :] = embedded.permute(
                        0, 2, 1
                    )  # [B, embedding_dim, T]

                # Mark categorical features as present in feature mask
                feature_mask[
                    :,
                    self.max_numeric_features : self.max_numeric_features
                    + num_categorical,
                    :,
                ] = 1.0

        return x_num_padded, x_cat_embedded, feature_mask

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

        # 1. Pad features to unified space with categorical embeddings
        x_num_padded, x_cat_embedded, feature_mask = self.pad_features_to_unified_space(
            x_num, x_cat
        )
        # x_num_padded: [B, max_numeric_features, T]
        # x_cat_embedded: [B, max_categorical_features, categorical_embedding_dim, T]
        # feature_mask: [B, max_numeric_features + max_categorical_features, T]

        # 2. Handle numeric features with dual-patch NaN handling
        m_nan = torch.isnan(x_num_padded).float()  # [B, max_numeric_features, T]
        x_num_val = torch.nan_to_num(
            x_num_padded, nan=0.0
        )  # [B, max_numeric_features, T]

        # 3. Prepare channels for concatenation
        channels_to_cat = []

        # Add numeric channels: [value, nan_mask, feature_mask?, column_emb?]
        channels_to_cat.append(x_num_val)  # [B, max_numeric_features, T]
        channels_to_cat.append(m_nan)  # [B, max_numeric_features, T]

        if self.use_feature_masks:
            numeric_feature_mask = feature_mask[
                :, : self.max_numeric_features, :
            ]  # [B, max_numeric_features, T]
            channels_to_cat.append(numeric_feature_mask)

        if self.use_column_embeddings and self.column_embedder is not None:
            col_emb = self.column_embedder(B, T)  # [B, max_total_features, T]
            numeric_col_emb = col_emb[
                :, : self.max_numeric_features, :
            ]  # [B, max_numeric_features, T]
            channels_to_cat.append(numeric_col_emb)

        # 4. Add categorical channels: [embedding_dims, feature_mask?, column_emb?]
        # Reshape categorical embeddings from
        # [B, max_categorical_features, embedding_dim, T] to
        # [B, max_categorical_features * embedding_dim, T]
        x_cat_reshaped = x_cat_embedded.view(
            B, self.max_categorical_features * self.categorical_embedding_dim, T
        )
        channels_to_cat.append(x_cat_reshaped)

        if self.use_feature_masks:
            categorical_feature_mask = feature_mask[
                :, self.max_numeric_features :, :
            ]  # [B, max_categorical_features, T]
            channels_to_cat.append(categorical_feature_mask)

        if self.use_column_embeddings and self.column_embedder is not None:
            categorical_col_emb = col_emb[
                :, self.max_numeric_features :, :
            ]  # [B, max_categorical_features, T]
            channels_to_cat.append(categorical_col_emb)

        # 5. Concatenate all channels
        x_processed = torch.cat(channels_to_cat, dim=1)  # [B, total_input_channels, T]

        # 6. Create patches
        patches = self.create_patches(x_processed)
        num_patches = patches.shape[1]

        # 7. Patch embedding
        patch_embeddings = self.patch_embedding(patches)  # [B, num_patches, d_model]

        # 8. Add positional encoding
        pos_enc = self.get_positional_encoding(num_patches)
        patch_embeddings = patch_embeddings + pos_enc

        # 9. Transformer
        transformer_out = self.transformer(
            patch_embeddings
        )  # [B, num_patches, d_model]

        # 10. Global average pooling
        pooled = transformer_out.mean(dim=1)  # [B, d_model]

        # 11. Classification/regression head
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
        categorical_cardinalities: list[int] | None = None,
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
            categorical_cardinalities: List of cardinality for each categorical feature
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
            categorical_cardinalities=categorical_cardinalities,
            **kwargs,
        )

        return model

    @classmethod
    def create_baseline(
        cls,
        max_numeric_features: int,
        max_categorical_features: int,
        num_classes: int,
        categorical_cardinalities: list[int] | None = None,
        **kwargs,
    ) -> "VariableFeaturePatchDuET":
        """Create baseline model without column embeddings."""
        return cls(
            max_numeric_features=max_numeric_features,
            max_categorical_features=max_categorical_features,
            num_classes=num_classes,
            column_embedding_strategy="none",
            categorical_cardinalities=categorical_cardinalities,
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
    # Define categorical cardinalities (max across all datasets)
    categorical_cardinalities = [
        10,
        5,
    ]  # 10 categories for first categorical, 5 for second

    model = VariableFeaturePatchDuET.create_for_multi_dataset(
        dataset_schemas=dataset_schemas,
        num_classes=3,
        strategy="auto_expanding",
        categorical_cardinalities=categorical_cardinalities,
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
