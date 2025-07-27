import polars as pol
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, Dataset
from duet.models import PatchTSTNan


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


# PatchTSTNan is now imported from duet.models


class W3DataModule(pl.LightningDataModule):
    def __init__(
        self, train_df: pol.DataFrame, batch_size: int = 32, sequence_length: int = 64
    ):
        super().__init__()
        self.train_df = train_df
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def setup(self, stage=None):
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

    # Create data module with maximum batch size for efficiency
    batch_size = (
        1024  # Large batch size possible due to reduced memory from large patches
    )
    data_module = W3DataModule(df, batch_size=batch_size, sequence_length=64)

    print(f"Training config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {len(df):,} rows")

    # Create model with optimal patch size for maximum speed
    patch_size = 32  # Optimal from benchmark: 8.39x speedup vs patch_size=8
    model = PatchTSTNan(
        C_num=C_num,
        C_cat=C_cat,
        cat_cardinality=1,  # Not used since C_cat=0
        T=64,
        d_model=128,
        patch_size=patch_size,  # 64/32 = 2 patches (vs 8 patches with size 8)
        num_classes=unique_states,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        learning_rate=1e-4,
    )

    print(f"Model config (optimized for speed):")
    print(f"  Sequence length: 64")
    print(f"  Patch size: {patch_size}")
    print(f"  Number of patches: {64 // patch_size}")
    print(f"  Effective transformer sequence length: {64 // patch_size}")
    print(f"  Expected speedup: ~8.4x vs patch_size=8")
    print(f"  Expected throughput: ~62 samples/second")
    print(f"  Batch size advantage: {batch_size} samples/batch")
    print(f"  Memory efficiency: ~{8 / 2}x less memory per sample vs patch_size=8")

    # Create trainer with TensorBoard logging
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger("tb_logs", name="patchtstnan_w3")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # Keep mixed precision for speed
        log_every_n_steps=50,  # Reduced logging frequency for speed
        logger=logger,
        gradient_clip_val=1.0,
        # Performance optimizations
        enable_checkpointing=True,
        enable_progress_bar=True,
        # Reduce validation frequency for speed during training
        val_check_interval=0.25,  # Check validation every 25% of epoch
        limit_val_batches=100,  # Limit validation to 100 batches for speed
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
