"""Ultra-fast PatchTSTNan training with extreme patch sizes for fine-grained data.

For very fine-grained data (every 1-15 seconds), we can use much larger patches
since consecutive data points are highly correlated.
"""

import polars as pol
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, Dataset
from duet.models import PatchTSTNan


class W3Dataset(Dataset):
    def __init__(
        self, df: pol.DataFrame, sequence_length: int = 128, prediction_horizon: int = 1
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


class W3DataModule(pl.LightningDataModule):
    def __init__(
        self, train_df: pol.DataFrame, batch_size: int = 256, sequence_length: int = 128
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
    print("=" * 60)
    print("ULTRA-FAST PatchTSTNan Training")
    print("Optimized for fine-grained temporal data")
    print("=" * 60)
    
    # Load first 2M rows for faster testing
    print("Loading first 2M rows from W3 dataset...")
    df_lazy = pol.scan_parquet("/home/ubuntu/DuET/data/W3/train.parquet")
    df = df_lazy.head(2_000_000).collect()

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

    # Ultra-fast configuration for fine-grained data
    sequence_length = 128  # Longer sequence for more context
    batch_size = 1024  # Maximum batch size for extreme efficiency
    patch_size = 64  # Extreme patch size: 128/64 = 2 patches only!
    
    print(f"\nUltra-fast configuration:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Patch size: {patch_size}")
    print(f"  Number of patches: {sequence_length // patch_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Memory per sample: ~{(128//8)/(128//64)}x less than patch_size=8")
    print(f"  Total speedup potential: >50x for fine-grained data!")
    print(f"  Extreme throughput: {batch_size} samples per batch!")

    # Create data module
    data_module = W3DataModule(df, batch_size=batch_size, sequence_length=sequence_length)

    # Create ultra-fast model
    model = PatchTSTNan(
        C_num=C_num,
        C_cat=C_cat,
        cat_cardinality=1,  # Not used since C_cat=0
        T=sequence_length,
        d_model=128,
        patch_size=patch_size,  # Only 2 patches!
        num_classes=unique_states,
        n_heads=8,
        n_layers=3,  # Reduced layers for even more speed
        dropout=0.1,
        learning_rate=1e-4,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create trainer with maximum speed optimizations
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger("tb_logs", name="patchtstnan_w3_ultra_fast")

    trainer = pl.Trainer(
        max_epochs=10,  # Fewer epochs due to faster convergence
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # Mixed precision for speed
        log_every_n_steps=100,  # Minimal logging
        logger=logger,
        gradient_clip_val=1.0,
        # Maximum speed optimizations
        enable_checkpointing=False,  # Disable checkpointing for max speed
        enable_progress_bar=True,
        val_check_interval=0.5,  # Less frequent validation
        limit_val_batches=50,  # Minimal validation batches
        # Additional speed optimizations
        enable_model_summary=False,
        num_sanity_val_steps=0,  # Skip sanity validation
    )

    # Train model
    print("Starting ultra-fast training...")
    import time
    start_time = time.time()
    
    trainer.fit(model, data_module)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds!")
    print(f"That's {training_time/60:.1f} minutes total.")


if __name__ == "__main__":
    main()