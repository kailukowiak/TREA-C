"""Quick test version of PatchTSTNan training with small dataset and optimizations."""

import polars as pol
import pytorch_lightning as pl
import torch
import time

from torch.utils.data import DataLoader, Dataset
from duet.models import PatchTSTNan


class W3Dataset(Dataset):
    def __init__(
        self, df: pol.DataFrame, sequence_length: int = 32, prediction_horizon: int = 1
    ):
        self.df = df
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon

        # Identify numeric and categorical columns
        self.numeric_cols = [
            col for col in df.columns if col not in ["state", "well_name"]
        ]
        self.categorical_cols = []

        # Convert to pandas for easier indexing
        self.data = df.to_pandas()

        # Pre-compute sequences for speed
        print("Pre-computing sequences...")
        start_time = time.time()
        
        self.sequences = []
        for well_name, group in self.data.groupby("well_name"):
            if len(group) >= self.seq_len + self.pred_horizon:
                # Limit sequences per well for speed (every 10th sequence)
                valid_starts = range(0, len(group) - self.seq_len - self.pred_horizon + 1, 10)
                for i in valid_starts:
                    self.sequences.append((well_name, i, i + self.seq_len))
        
        compute_time = time.time() - start_time
        print(f"Pre-computed {len(self.sequences)} sequences in {compute_time:.2f}s")

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
        self, train_df: pol.DataFrame, batch_size: int = 64, sequence_length: int = 32
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
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,  # Parallel data loading
            pin_memory=True  # Faster GPU transfer
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=4,
            pin_memory=True
        )


def main():
    print("=" * 60)
    print("QUICK TEST: PatchTSTNan on W3 Dataset")
    print("Small dataset + optimizations for fast testing")
    print("=" * 60)

    # Load SMALL subset for quick testing
    print("Loading first 200K rows from W3 dataset...")
    start_time = time.time()
    
    df_lazy = pol.scan_parquet("/home/ubuntu/DuET/data/W3/train.parquet")
    df = df_lazy.head(200_000).collect()  # Small for testing
    
    load_time = time.time() - start_time
    print(f"Loaded {len(df)} rows in {load_time:.2f}s")

    # Data preprocessing
    print("\nPreprocessing data...")
    df = df.drop("class").filter(pol.col("state").is_not_null())

    # Convert state to string then integer
    df = df.with_columns([pol.col("state").cast(pol.Utf8)])
    
    # Remove rows where all numeric values are null
    numeric_cols = [col for col in df.columns if col not in ["state", "well_name"]]
    df = df.filter(
        ~pol.all_horizontal([pol.col(col).is_null() for col in numeric_cols])
    )

    print(f"After preprocessing: {len(df)} rows")
    print(f"Wells: {df['well_name'].n_unique()}")
    print(f"States: {df['state'].value_counts()}")

    # Get dataset parameters
    C_num = len(numeric_cols)
    unique_states = df["state"].unique().len()

    # Create state label mapping
    state_labels = df["state"].unique().sort().to_list()
    state_to_label = {state: i for i, state in enumerate(state_labels)}
    print(f"State mapping: {state_to_label}")

    # Convert states to integer labels
    df = df.with_columns(
        [
            pol.col("state").map_elements(
                lambda x: state_to_label[x], return_dtype=pol.Int64
            )
        ]
    )

    # Quick test configuration
    sequence_length = 32  # Short sequences
    batch_size = 64       # Moderate batch size
    patch_size = 16       # 32/16 = 2 patches

    print(f"\nQuick test configuration:")
    print(f"  Dataset size: {len(df):,} rows")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Features: {C_num}")
    print(f"  Classes: {unique_states}")

    # Create data module
    print(f"\nCreating data module...")
    data_module = W3DataModule(df, batch_size=batch_size, sequence_length=sequence_length)

    # Create model
    model = PatchTSTNan(
        C_num=C_num,
        C_cat=0,
        T=sequence_length,
        d_model=64,  # Smaller for testing
        patch_size=patch_size,
        num_classes=unique_states,
        n_heads=4,   # Fewer heads for speed
        n_layers=2,  # Fewer layers for speed
        dropout=0.1,
        learning_rate=1e-3,  # Higher LR for faster convergence
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test data loading speed
    print(f"\nTesting data loading...")
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    
    print(f"Number of batches: {len(train_loader)}")
    
    # Time a few batches
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Test first 3 batches
            break
        print(f"  Batch {i+1}: x_num {batch['x_num'].shape}, y {batch['y'].shape}")
    
    data_time = time.time() - start_time
    print(f"Data loading time for 3 batches: {data_time:.2f}s ({data_time/3:.2f}s per batch)")

    # Test model forward pass
    print(f"\nTesting model forward pass...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        
        start_time = time.time()
        output = model(sample_batch['x_num'], sample_batch['x_cat'])
        forward_time = time.time() - start_time
        
        print(f"Forward pass time: {forward_time:.3f}s")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Contains NaN: {torch.isnan(output).any()}")
        
        # Test loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, sample_batch['y'])
        print(f"Sample loss: {loss.item():.4f}")
        print(f"Loss is NaN: {torch.isnan(loss)}")

    # Quick training test
    print(f"\nStarting quick training test...")
    
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("tb_logs", name="patchtstnan_w3_quick_test")

    trainer = pl.Trainer(
        max_epochs=2,  # Just 2 epochs for testing
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        logger=logger,
        limit_train_batches=20,  # Only 20 batches per epoch for testing
        limit_val_batches=5,     # Only 5 validation batches
        enable_checkpointing=False,  # No checkpointing for speed
        num_sanity_val_steps=0,      # Skip sanity check
    )

    # Time the training
    total_start_time = time.time()
    trainer.fit(model, data_module)
    total_time = time.time() - total_start_time

    print(f"\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Time per epoch: {total_time/2:.2f}s")
    print(f"Time per batch: {total_time/40:.2f}s")  # 20 batches * 2 epochs

    if trainer.callback_metrics:
        print("\nFinal metrics:")
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")

    print(f"\nScaling estimates for full dataset:")
    full_batches = 1523  # From your original run
    estimated_time_per_epoch = (total_time/40) * full_batches
    print(f"  Estimated time per epoch: {estimated_time_per_epoch/3600:.1f} hours")
    print(f"  Recommended: Use smaller dataset or increase optimizations!")


if __name__ == "__main__":
    main()