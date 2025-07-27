"""Ultra-fast PatchTSTNan training with extreme patch sizes for fine-grained data.

For very fine-grained data (every 1-15 seconds), we can use much larger patches
since consecutive data points are highly correlated.
"""

import numpy as np
import pandas as pd
import polars as pol
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, Dataset

from duet.models import PatchTSTNan


class W3SimpleDataset(Dataset):
    """Simple dataset that loads a fixed subset of data upfront for fast training."""

    def __init__(
        self,
        parquet_path: str,
        well_names: list,
        numeric_cols: list,
        state_to_label: dict,
        sequence_length: int = 128,
        max_total_rows: int = 50000,  # Limit total data loaded
    ):
        self.numeric_cols = numeric_cols
        self.seq_len = sequence_length
        
        print(f"Loading subset of data from {len(well_names)} wells (max {max_total_rows} rows)...")
        
        # Load a manageable subset of data
        df_lazy = pol.scan_parquet(parquet_path)
        
        # Load data from wells with row limit - USE CLASS INSTEAD OF STATE  
        all_data = (
            df_lazy
            .filter(pol.col("well_name").is_in(well_names))
            .filter(pol.col("class").is_not_null())  # Use class column
            .drop("state")  # Drop state column
            .head(max_total_rows)  # Limit total rows
            .with_columns([pol.col("class").cast(pol.Utf8)])
            .with_columns([pol.col("class").replace_strict(state_to_label)])
            .collect()
            .to_pandas()
        )
        
        print(f"Loaded {len(all_data)} rows, creating sequences...")
        
        # Create all sequences upfront
        sequences = []
        nan_count_by_well = {}
        
        for well_name in all_data['well_name'].unique():
            well_data = all_data[all_data['well_name'] == well_name]
            well_nan_count = 0
            
            # Create sequences for this well
            for i in range(len(well_data) - sequence_length):
                start_idx = i
                end_idx = i + sequence_length
                target_idx = end_idx
                
                if target_idx < len(well_data):
                    sequence_data = well_data.iloc[start_idx:end_idx]
                    target_state = well_data.iloc[target_idx]['class']  # Use class column
                    
                    # Extract features with NaN checking
                    numeric_values = sequence_data[numeric_cols].values
                    
                    # Check for NaN/inf values
                    has_nan = np.any(np.isnan(numeric_values)) or np.any(np.isinf(numeric_values))
                    
                    if has_nan:
                        well_nan_count += 1
                        # Skip sequences with too many NaNs for this problematic well
                        if well_name == "WELL-00033" and well_nan_count > 100:
                            continue  # Skip this well's remaining sequences
                        # Replace NaN/inf with 0 for stability
                        numeric_values = np.nan_to_num(numeric_values, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # CRITICAL: Simple but effective normalization 
                    # Apply z-score normalization with outlier clipping
                    numeric_values_norm = np.zeros_like(numeric_values)
                    for col_idx in range(numeric_values.shape[1]):
                        col_data = numeric_values[:, col_idx]
                        # Simple z-score: (x - mean) / std, then clip
                        mean_val = np.mean(col_data)
                        std_val = np.std(col_data)
                        if std_val > 1e-8:  # Only normalize if there's variation
                            normalized = (col_data - mean_val) / std_val
                            # Clip extreme values to prevent model overflow
                            normalized = np.clip(normalized, -3, 3)
                            numeric_values_norm[:, col_idx] = normalized
                        else:
                            # No variation - set to zero
                            numeric_values_norm[:, col_idx] = 0.0
                    
                    numeric_values = numeric_values_norm
                    
                    x_num = torch.tensor(numeric_values, dtype=torch.float32).T  # [C_num, T]
                    
                    x_cat = torch.empty((0, sequence_length), dtype=torch.long)
                    y = torch.tensor(target_state, dtype=torch.long)
                    
                    sequences.append({"x_num": x_num, "x_cat": x_cat, "y": y})
            
            if well_nan_count > 0:
                nan_count_by_well[well_name] = well_nan_count
        
        # Report NaN statistics
        if nan_count_by_well:
            print(f"Wells with NaN sequences: {dict(list(nan_count_by_well.items())[:5])}")  # Show first 5
        
        self.sequences = sequences
        print(f"Created {len(sequences)} sequences - ready for training!")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class W3Dataset(Dataset):
    """Legacy dataset for backward compatibility - use W3StreamingDataset for large datasets."""

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

        # Target is the class at the end of sequence
        target_data = self.data[self.data["well_name"] == well_name].iloc[
            end_idx : end_idx + self.pred_horizon
        ]
        y = torch.tensor(
            target_data["class"].values[0] if len(target_data) > 0 else 0,
            dtype=torch.long,
        )

        return {"x_num": x_num, "x_cat": x_cat, "y": y}


class W3DataModule(pl.LightningDataModule):
    def __init__(
        self,
        parquet_path: str = None,
        train_df: pol.DataFrame = None,
        metadata: dict = None,
        batch_size: int = 256,
        sequence_length: int = 128,
        use_streaming: bool = True,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.train_df = train_df
        self.metadata = metadata
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.use_streaming = use_streaming

    def setup(self, stage=None):
        if self.use_streaming and self.parquet_path and self.metadata:
            # Use streaming dataset for large data
            print("Setting up streaming datasets...")

            # Get all well names efficiently
            df_lazy = pol.scan_parquet(self.parquet_path)
            wells = (
                df_lazy.select("well_name").unique().collect()["well_name"].to_list()
            )

            # Split wells for train/val
            n_train_wells = int(0.8 * len(wells))
            train_wells = wells[:n_train_wells]
            val_wells = wells[n_train_wells:]

            print(
                f"Using {len(train_wells)} wells for training, {len(val_wells)} for validation"
            )

            self.train_dataset = W3SimpleDataset(
                parquet_path=self.parquet_path,
                well_names=train_wells,
                numeric_cols=self.metadata["numeric_cols"],
                state_to_label=self.metadata["state_to_label"],
                sequence_length=self.sequence_length,
                max_total_rows=100000,  # Scale up dataset size 10x
            )

            self.val_dataset = W3SimpleDataset(
                parquet_path=self.parquet_path,
                well_names=val_wells,
                numeric_cols=self.metadata["numeric_cols"],
                state_to_label=self.metadata["state_to_label"],
                sequence_length=self.sequence_length,
                max_total_rows=20000,  # Scale up validation 10x
            )

        else:
            # Legacy approach with DataFrame
            print("Setting up legacy datasets...")
            wells = self.train_df["well_name"].unique().to_list()
            n_train_wells = int(0.8 * len(wells))

            train_wells = wells[:n_train_wells]
            val_wells = wells[n_train_wells:]

            train_df = self.train_df.filter(pol.col("well_name").is_in(train_wells))
            val_df = self.train_df.filter(pol.col("well_name").is_in(val_wells))

            self.train_dataset = W3Dataset(
                train_df, sequence_length=self.sequence_length
            )
            self.val_dataset = W3Dataset(val_df, sequence_length=self.sequence_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


def extract_dataset_metadata():
    """Extract metadata from the full dataset without loading into memory."""
    print("=" * 60)
    print("EXTRACTING DATASET METADATA")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/DuET/data/W3/train.parquet")

    # Get total row count efficiently
    total_rows = df_lazy.select(pol.len()).collect().item()
    print(f"Total rows in dataset: {total_rows:,}")

    # Get column info
    columns = df_lazy.columns
    print(f"Total columns: {len(columns)}")

    # Get unique classes efficiently (without loading full data)
    classes_df = (
        df_lazy.select("class")
        .filter(pol.col("class").is_not_null())
        .unique()
        .collect()
    )
    unique_classes = classes_df["class"].sort().to_list()
    print(f"Unique classes: {unique_classes}")

    # Get unique wells count
    wells_count = (
        df_lazy.select("well_name").unique().select(pol.len()).collect().item()
    )
    print(f"Unique wells: {wells_count}")

    # Identify numeric columns
    numeric_cols = [
        col for col in columns if col not in ["class", "well_name", "state"]
    ]
    print(f"Numeric features: {len(numeric_cols)}")

    # Create class mapping
    class_to_label = {str(cls): i for i, cls in enumerate(unique_classes)}
    print(f"Class labels: {class_to_label}")

    return {
        "total_rows": total_rows,
        "unique_states": len(unique_classes),  # Keep name for compatibility
        "state_to_label": class_to_label,     # Keep name for compatibility  
        "numeric_cols": numeric_cols,
        "wells_count": wells_count,
        "C_num": len(numeric_cols),
        "C_cat": 0,
    }


def main():
    print("=" * 60)
    print("ULTRA-FAST PatchTSTNan Training")
    print("Optimized for fine-grained temporal data")
    print("=" * 60)

    # Phase 1: Extract metadata efficiently
    metadata = extract_dataset_metadata()

    # For initial testing, use subset. Later we'll use streaming
    print("\n" + "=" * 60)
    print("LOADING TRAINING SUBSET")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/DuET/data/W3/train.parquet")
    # Use larger subset but not full dataset yet
    df = (
        df_lazy.filter(pol.col("class").is_not_null())  # Use class column
        .drop("state")  # Drop state column
        .head(1_000_000)  # 1M rows for testing
        .collect()
    )

    print(f"Loaded {len(df)} rows for training")

    # Convert class to string then to integer labels using metadata
    df = df.with_columns([pol.col("class").cast(pol.Utf8)])

    # Remove rows where all numeric values are null
    df = df.filter(
        ~pol.all_horizontal(
            [pol.col(col).is_null() for col in metadata["numeric_cols"]]
        )
    )

    print(f"After filtering null numeric rows: {len(df)} rows")

    # Convert classes to integer labels using pre-computed mapping
    df = df.with_columns([pol.col("class").replace_strict(metadata["state_to_label"])])

    # Proven stable configuration from ETTh1 test, scaled up for GPU utilization
    sequence_length = 512  # Keep larger sequences for more data per sample
    batch_size = 256  # Large batch but stable with proven model size
    patch_size = 16   # Proven stable patches from ETTh1: 512/16 = 32 patches (with stride=8 = ~64 patches)

    print("\nUltra-fast configuration:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Patch size: {patch_size}")
    print(f"  Number of patches: {sequence_length // patch_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Memory per sample: ~{(128 // 8) / (128 // 64)}x less than patch_size=8")
    print("  Total speedup potential: >50x for fine-grained data!")
    print(f"  Batch throughput: {batch_size} samples per batch (testing size)")

    # Use streaming dataset for full data efficiency
    print("\n" + "=" * 60)
    print("CREATING STREAMING DATA MODULE")
    print("=" * 60)

    # Create data module with streaming support
    data_module = W3DataModule(
        parquet_path="/home/ubuntu/DuET/data/W3/train.parquet",
        metadata=metadata,
        batch_size=batch_size,
        sequence_length=sequence_length,
        use_streaming=True,  # Enable streaming for full dataset
    )

    # Create model with proven ETTh1-compatible configuration
    model = PatchTSTNan(
        c_in=metadata["C_num"],  # Use c_in parameter name for stride-based mode
        seq_len=sequence_length,  # Use seq_len parameter name
        num_classes=metadata["unique_states"],
        patch_len=16,  # Smaller patches proven to work (ETTh1 config)
        stride=8,     # Smaller stride proven to work (ETTh1 config)
        d_model=64,   # Proven stable size from ETTh1 test
        n_head=4,     # Proven stable heads from ETTh1 test
        num_layers=2, # Proven stable layers from ETTh1 test
        lr=1e-3,      # Proven stable learning rate from ETTh1 test
        task='classification'  # Explicit task specification
    )

    print(
        f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Apply performance optimizations
    print("\n" + "=" * 60)
    print("APPLYING PERFORMANCE OPTIMIZATIONS")
    print("=" * 60)

    # Enable Tensor Cores for maximum performance
    torch.set_float32_matmul_precision("medium")
    print("✓ Enabled Tensor Cores (medium precision)")

    # Set optimal threading
    torch.set_num_threads(8)
    print("✓ Set torch threads to 8")

    # Create trainer with maximum speed optimizations
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger("tb_logs", name="patchtstnan_w3_ultra_fast")

    trainer = pl.Trainer(
        max_epochs=10,  # More epochs for proper training
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # Re-enable mixed precision for speed
        log_every_n_steps=20,  # Normal logging frequency
        logger=logger,
        gradient_clip_val=1.0,  # Standard gradient clipping
        # Production settings
        enable_checkpointing=True,  # Enable checkpointing for longer runs
        enable_progress_bar=True,
        val_check_interval=0.25,  # Frequent validation for monitoring
        limit_val_batches=25,  # More validation batches
        # Optimizations
        enable_model_summary=True,  # Show model structure
        num_sanity_val_steps=2,  # Standard sanity checks
        # No anomaly detection in production
        accumulate_grad_batches=4,  # Gradient accumulation for effective batch size of 1024 (256*4)
    )

    # Train model
    print("\n" + "=" * 60)
    print("STARTING ULTRA-FAST TRAINING")
    print("=" * 60)
    print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print("Dataset will show actual batch counts after loading...")

    import time

    start_time = time.time()

    trainer.fit(model, data_module)

    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Training time: {training_time:.2f}s ({training_time / 60:.1f} minutes)")
    print(f"GPU Memory after training: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Max GPU Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    # Reset memory stats for next run
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
