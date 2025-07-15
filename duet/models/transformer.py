"""Dual-Patch Transformer model implementation."""

import pytorch_lightning as pl
import torch
import torch.nn as nn


class DualPatchTransformer(pl.LightningModule):
    """Dual-Patch Transformer for time series with numeric and categorical features.

    Handles missing values efficiently by encoding them in a dual-patch format.
    """

    def __init__(
        self,
        C_num: int,
        C_cat: int,
        cat_cardinalities: list[int],
        T: int,
        d_model: int = 64,
        task: str = "classification",
        num_classes: int | None = 3,
        nhead: int = 4,
        num_layers: int = 2,
        lr: float = 1e-3,
        pooling: str = "mean",
        dropout: float = 0.1,
    ):
        """Initialize the Dual-Patch Transformer.

        Args:
            C_num: Number of numeric channels
            C_cat: Number of categorical channels
            cat_cardinalities: List of unique values per categorical channel
            T: Number of time steps
            d_model: Model dimension
            task: 'classification' or 'regression'
            num_classes: Number of classes for classification
                (required if task='classification')
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            lr: Learning rate
            pooling: Pooling strategy ('mean', 'last', 'cls')
            dropout: Dropout rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.d_model = d_model
        self.lr = lr
        self.pooling = pooling
        self.T = T

        # Validate inputs
        if len(cat_cardinalities) != C_cat:
            raise ValueError(
                f"Length of cat_cardinalities ({len(cat_cardinalities)}) "
                f"must match C_cat ({C_cat})"
            )
        if task == "classification" and num_classes is None:
            raise ValueError("num_classes must be specified for classification task")
        if pooling not in ["mean", "last", "cls"]:
            raise ValueError(
                f"pooling must be one of ['mean', 'last', 'cls'], got {pooling}"
            )

        # Categorical embeddings (time-varying) with variable cardinalities
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities]
        )

        # Numeric projection: NOTE 2× input channels for value + mask
        self.num_proj = nn.Conv1d(C_num * 2, d_model, kernel_size=1)

        # Add CLS token if using cls pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Temporal encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output heads
        if task == "classification":
            self.head = nn.Linear(d_model, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.MSELoss()

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_num: Numeric features [B, C_num, T]
            x_cat: Categorical features [B, C_cat, T]

        Returns:
            Model output [B, num_classes] or [B, 1]
        """
        # 1-2. Always use dual-patch: mask + zero-fill (consistent shape)
        m_nan = torch.isnan(x_num).float()  # [B, C_num, T]
        x_val = torch.nan_to_num(x_num, nan=0.0)  # [B, C_num, T]

        # 3. stack value & mask channels (always dual-patch)
        x_num2 = torch.cat([x_val, m_nan], dim=1)  # [B, 2·C_num, T]

        # 4. project numeric features
        z_num = self.num_proj(x_num2)  # [B, d_model, T]

        # 5. add categorical embeddings (safe summation)
        if len(self.cat_embs) > 0:
            cat_vecs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
            cat_vec = torch.stack(cat_vecs, dim=0).sum(dim=0)  # [B, T, d_model]
            z = z_num + cat_vec.permute(0, 2, 1)  # align dims: [B, d_model, T]
        else:
            z = z_num

        # 6. Add CLS token if using cls pooling
        z = z.permute(0, 2, 1)  # [B, T, d_model]
        if self.pooling == "cls":
            batch_size = z.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
            z = torch.cat([cls_tokens, z], dim=1)  # [B, T+1, d_model]

        # 7. transformer + configurable pooling
        z = self.transformer(z)

        if self.pooling == "mean":
            if hasattr(self, "cls_token"):
                z = z[:, 1:, :].mean(dim=1)  # exclude CLS token
            else:
                z = z.mean(dim=1)  # global avg pooling
        elif self.pooling == "last":
            z = z[:, -1, :]  # last time step
        elif self.pooling == "cls":
            z = z[:, 0, :]  # CLS token

        return self.head(z)

    def training_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Training step."""
        out = self(batch["x_num"], batch["x_cat"])
        loss = self.loss_fn(out.squeeze(), batch["y"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Validation step."""
        out = self(batch["x_num"], batch["x_cat"])
        loss = self.loss_fn(out.squeeze(), batch["y"])
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create model from DatasetConfig.

        Args:
            config: DatasetConfig instance
            **kwargs: Additional model parameters

        Returns:
            DualPatchTransformer instance
        """
        model_params = config.get_model_params()
        model_params.update(kwargs)
        return cls(**model_params)
