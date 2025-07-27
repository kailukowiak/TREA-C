import polars as pol
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset


class W3Dataset(Dataset):
    def __init__(
        self, df: pol.DataFrame, sequence_length: int = 64, prediction_horizon: int = 1
    ):
        self.df = df
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon

        # Identify numeric and categorical columns
        self.numeric_cols = [
            col for col in df.columns if col not in ["state", "well_name"]
        ]
        self.categorical_cols = []  # No categorical features, state is now the target

        # Convert to pandas for easier indexing
        self.data = df.to_pandas()

        # Group by well_name to create sequences
        self.sequences = []
        for well_name, group in self.data.groupby("well_name"):
            if len(group) >= self.seq_len + self.pred_horizon:
                for i in range(len(group) - self.seq_len - self.pred_horizon + 1):
                    self.sequences.append((well_name, i, i + self.seq_len))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        well_name, start_idx, end_idx = self.sequences[idx]
        well_data = self.data[self.data["well_name"] == well_name].iloc[
            start_idx:end_idx
        ]

        # Extract numeric features
        x_num = torch.tensor(
            well_data[self.numeric_cols].values, dtype=torch.float32
        ).T  # [C_num, T]

        # No categorical features
        x_cat = torch.empty((0, self.seq_len), dtype=torch.long)

        # Target is the state at the end of sequence
        target_data = self.data[self.data["well_name"] == well_name].iloc[
            end_idx : end_idx + self.pred_horizon
        ]
        y = torch.tensor(
            target_data["state"].values[0] if len(target_data) > 0 else 0,
            dtype=torch.long,
        )

        return {"x_num": x_num, "x_cat": x_cat, "y": y}


class PatchTSTNan(pl.LightningModule):
    def __init__(
        self,
        C_num: int,
        C_cat: int,
        cat_cardinality: int,
        T: int,
        d_model: int = 64,
        patch_size: int = 8,
        num_classes: int = 2,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.C_num = C_num
        self.C_cat = C_cat
        self.T = T
        self.patch_size = patch_size
        self.n_patches = T // patch_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Categorical embeddings
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(cat_cardinality, d_model) for _ in range(C_cat)]
        )

        # Numeric projection with missing value handling (double channels for mask)
        self.num_proj = nn.Conv1d(C_num * 2, d_model, kernel_size=1)

        # Patch embedding
        self.patch_proj = nn.Linear(patch_size, d_model)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Linear(d_model, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def create_patches(self, x):
        """Convert time series to patches"""
        B, C, T = x.shape
        # Reshape to patches: [B, C, n_patches, patch_size]
        x_patches = x.view(B, C, self.n_patches, self.patch_size)
        # Flatten spatial dimension: [B, C * n_patches, patch_size]
        x_patches = x_patches.view(B, C * self.n_patches, self.patch_size)
        return x_patches

    def forward(self, x_num, x_cat):
        _B = x_num.shape[0]

        # Handle missing values in numeric data
        m_nan = torch.isnan(x_num).float()  # [B, C_num, T]
        x_val = torch.nan_to_num(x_num, nan=0.0)  # [B, C_num, T]

        # Stack value and mask channels
        x_num_with_mask = torch.cat([x_val, m_nan], dim=1)  # [B, 2*C_num, T]

        # Project numeric features
        z_num = self.num_proj(x_num_with_mask)  # [B, d_model, T]

        # Add categorical embeddings (skip if no categorical features)
        if self.C_cat > 0 and x_cat.numel() > 0:
            cat_emb = sum(emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs))
            z_num = z_num + cat_emb.permute(0, 2, 1)

        # Create patches
        z_patches = self.create_patches(z_num)  # [B, C*n_patches, patch_size]

        # Project patches to model dimension
        z_patches = self.patch_proj(z_patches)  # [B, C*n_patches, d_model]

        # Add positional embeddings (broadcast across channels)
        n_channel_patches = z_patches.shape[1]
        pos_embed_expanded = self.pos_embed.repeat(
            1, n_channel_patches // self.n_patches, 1
        )
        z_patches = z_patches + pos_embed_expanded

        # Apply dropout
        z_patches = self.dropout(z_patches)

        # Transformer encoding
        z_encoded = self.transformer(z_patches)  # [B, C*n_patches, d_model]

        # Global average pooling
        z_pooled = z_encoded.mean(dim=1)  # [B, d_model]

        # Classification
        logits = self.head(z_pooled)

        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["x_num"], batch["x_cat"])

        # Debug prints for first few batches
        if batch_idx < 3:
            print(f"Batch {batch_idx}:")
            print(f"  Logits shape: {logits.shape}, Logits: {logits}")
            print(f"  Target shape: {batch['y'].shape}, Targets: {batch['y']}")
            print(f"  Logits contains NaN: {torch.isnan(logits).any()}")
            print(f"  Logits contains Inf: {torch.isinf(logits).any()}")

        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WARNING: NaN/Inf detected in logits at batch {batch_idx}")
            return None

        loss = self.loss_fn(logits, batch["y"])

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss at batch {batch_idx}")
            return None

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["y"]).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["x_num"], batch["x_cat"])
        loss = self.loss_fn(logits, batch["y"])

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["y"]).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]


class W3DataModule(pl.LightningDataModule):
    def __init__(
        self, train_df: pol.DataFrame, batch_size: int = 32, sequence_length: int = 64
    ):
        super().__init__()
        self.train_df = train_df
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def setup(self, stage: str = None):
        # Split data by wells for train/val
        wells = self.train_df["well_name"].unique().to_list()
        n_train_wells = int(0.8 * len(wells))

        train_wells = wells[:n_train_wells]
        val_wells = wells[n_train_wells:]

        train_df = self.train_df.filter(pol.col("well_name").is_in(train_wells))
        val_df = self.train_df.filter(pol.col("well_name").is_in(val_wells))

        self.train_dataset = W3Dataset(train_df, sequence_length=self.sequence_length)
        self.val_dataset = W3Dataset(val_df, sequence_length=self.sequence_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)


def main():
    # Load first 5M rows
    print("Loading first 5M rows from W3 dataset...")
    df_lazy = pol.scan_parquet("/home/ubuntu/DuET/data/W3/train.parquet")
    df = df_lazy.head(5_000_000).collect()

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns}")

    # Drop class column and drop null state rows
    df = df.drop("class").filter(pol.col("state").is_not_null())

    print(f"After filtering null state rows: {len(df)} rows")
    print(f"Original state values: {df['state'].value_counts()}")

    # Convert state to string (categorical)
    df = df.with_columns([pol.col("state").cast(pol.Utf8)])

    # Remove rows where all numeric values are null
    numeric_cols = [col for col in df.columns if col not in ["state", "well_name"]]
    df = df.filter(
        ~pol.all_horizontal([pol.col(col).is_null() for col in numeric_cols])
    )

    print(f"After filtering null numeric rows: {len(df)} rows")

    # Get dataset parameters
    C_num = len(numeric_cols)
    C_cat = 0  # No categorical features, state is target
    unique_states = df["state"].unique().len()

    print("Dataset parameters:")
    print(f"  Numeric features: {C_num}")
    print(f"  Categorical features: {C_cat}")
    print(f"  Unique states: {unique_states}")

    # Create state label mapping (string to integer)
    state_labels = df["state"].unique().sort().to_list()
    state_to_label = {state: i for i, state in enumerate(state_labels)}
    print(f"  State labels: {state_to_label}")

    # Convert states to integer labels
    df = df.with_columns(
        [
            pol.col("state").map_elements(
                lambda x: state_to_label[x], return_dtype=pol.Int64
            )
        ]
    )

    # Create data module
    data_module = W3DataModule(df, batch_size=64, sequence_length=64)

    # Create model
    model = PatchTSTNan(
        C_num=C_num,
        C_cat=C_cat,
        cat_cardinality=1,  # Not used since C_cat=0
        T=64,
        d_model=128,
        patch_size=8,
        num_classes=unique_states,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        learning_rate=1e-4,
    )

    # Create trainer with TensorBoard logging
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger("tb_logs", name="patchtstnan_w3")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=1,
        logger=logger,
        gradient_clip_val=1.0,
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
