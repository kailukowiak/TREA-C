"""Train PatchTSTNan with the improved diverse W3 dataset."""

import polars as pol
import pytorch_lightning as pl
import torch
import time

from torch.utils.data import DataLoader, Dataset
from duet.models import PatchTSTNan


class ImprovedW3Dataset(Dataset):
    """Optimized dataset class for the improved W3 data."""
    
    def __init__(
        self, df: pol.DataFrame, sequence_length: int = 64, prediction_horizon: int = 1
    ):
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon

        # Identify columns
        self.numeric_cols = [
            col for col in df.columns if col not in ["state", "well_name"]
        ]
        
        print(f"Dataset initialization:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Numeric features: {len(self.numeric_cols)}")
        print(f"  Wells: {df['well_name'].n_unique()}")
        print(f"  States: {df['state'].unique().to_list()}")

        # Convert to pandas for faster indexing
        self.data = df.to_pandas()

        # Pre-compute sequences efficiently
        print("Pre-computing sequences...")
        start_time = time.time()
        
        self.sequences = []
        
        # Group by well and create sequences
        for well_name, group in self.data.groupby("well_name"):
            group_len = len(group)
            if group_len >= self.seq_len + self.pred_horizon:
                # For large wells, subsample to avoid memory issues
                if group_len > 10000:
                    # Take every Nth sequence for very large wells
                    step = max(1, group_len // 5000)  # Max 5000 sequences per well
                    valid_starts = range(0, group_len - self.seq_len - self.pred_horizon + 1, step)
                else:
                    # For smaller wells, take every 5th sequence
                    valid_starts = range(0, group_len - self.seq_len - self.pred_horizon + 1, 5)
                
                for start_idx in valid_starts:
                    self.sequences.append((well_name, start_idx, start_idx + self.seq_len))
        
        compute_time = time.time() - start_time
        print(f"Pre-computed {len(self.sequences):,} sequences in {compute_time:.2f}s")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        well_name, start_idx, end_idx = self.sequences[idx]
        well_data = self.data[self.data["well_name"] == well_name].iloc[start_idx:end_idx]

        # Extract numeric features
        x_num = torch.tensor(
            well_data[self.numeric_cols].values, dtype=torch.float32
        ).T  # [C_num, T]

        # No categorical features for W3
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


class ImprovedW3DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        df: pol.DataFrame, 
        batch_size: int = 512, 
        sequence_length: int = 64,
        num_workers: int = 8
    ):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Split by wells for train/val to avoid data leakage
        wells = self.df["well_name"].unique().to_list()
        n_train_wells = int(0.8 * len(wells))

        # Randomize well assignment
        import random
        random.shuffle(wells)
        
        train_wells = wells[:n_train_wells]
        val_wells = wells[n_train_wells:]

        print(f"Data split:")
        print(f"  Train wells: {len(train_wells)}")
        print(f"  Val wells: {len(val_wells)}")

        train_df = self.df.filter(pol.col("well_name").is_in(train_wells))
        val_df = self.df.filter(pol.col("well_name").is_in(val_wells))

        print(f"  Train rows: {len(train_df):,}")
        print(f"  Val rows: {len(val_df):,}")

        self.train_dataset = ImprovedW3Dataset(train_df, sequence_length=self.sequence_length)
        self.val_dataset = ImprovedW3Dataset(val_df, sequence_length=self.sequence_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


def main():
    print("=" * 60)
    print("IMPROVED W3 PATCHTSTNAN TRAINING")
    print("Using diverse 5M dataset with optimizations")
    print("=" * 60)

    # Configuration
    USE_TEST_DATASET = True  # Set to False for full 5M dataset
    
    if USE_TEST_DATASET:
        dataset_path = "/home/ubuntu/DuET/data/W3/improved_train_5M_test_500K.parquet"
        print("Using TEST dataset (500K rows) for validation")
    else:
        dataset_path = "/home/ubuntu/DuET/data/W3/improved_train_5M.parquet"
        print("Using FULL dataset (5M rows)")

    # Load improved dataset
    print(f"Loading dataset from: {dataset_path}")
    try:
        df = pol.read_parquet(dataset_path)
        print(f"✅ Loaded {len(df):,} rows")
    except FileNotFoundError:
        print(f"❌ Dataset not found. Please run: create_diverse_w3_dataset.py first")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Analyze loaded data
    print(f"\nDataset analysis:")
    print(f"  Rows: {len(df):,}")
    print(f"  Wells: {df['well_name'].n_unique()}")
    print(f"  States: {df['state'].value_counts()}")

    # Check for multi-class
    unique_states = df["state"].n_unique()
    if unique_states < 2:
        print("❌ Still only 1 class! Dataset creation failed.")
        print("Try running create_diverse_w3_dataset.py again")
        return

    print(f"✅ Found {unique_states} classes - ready for training!")

    # Prepare data
    numeric_cols = [col for col in df.columns if col not in ["state", "well_name"]]
    C_num = len(numeric_cols)

    # Create state mapping
    state_labels = df["state"].unique().sort().to_list()
    state_to_label = {state: i for i, state in enumerate(state_labels)}
    print(f"State mapping: {state_to_label}")

    # Convert states to integer labels
    df = df.with_columns([
        pol.col("state").replace_strict(state_to_label)
    ])

    # Optimized configuration for speed
    sequence_length = 64
    batch_size = 1024  # Large batch for efficiency
    patch_size = 32    # Optimized patch size (64/32 = 2 patches)
    num_workers = 8    # Parallel data loading

    print(f"\nOptimized configuration:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patch size: {patch_size} (2 patches)")
    print(f"  Features: {C_num}")
    print(f"  Classes: {unique_states}")
    print(f"  Num workers: {num_workers}")

    # Create data module
    print(f"\nCreating data module...")
    data_module = ImprovedW3DataModule(
        df, 
        batch_size=batch_size, 
        sequence_length=sequence_length,
        num_workers=num_workers
    )

    # Create optimized model
    model = PatchTSTNan(
        C_num=C_num,
        C_cat=0,
        T=sequence_length,
        d_model=128,
        patch_size=patch_size,
        num_classes=unique_states,
        n_heads=8,
        n_layers=3,  # Reduced for speed
        dropout=0.1,
        learning_rate=1e-3,
        task='classification'
    )

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {model_params:,}")

    # Test data loading speed
    print(f"\nTesting data loading speed...")
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    
    print(f"Number of batches: {len(train_loader):,}")
    
    # Time first few batches
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"  Batch {i+1}: x_num {batch['x_num'].shape}, y unique: {torch.unique(batch['y'])}")
    
    data_time = time.time() - start_time
    avg_batch_time = data_time / 3
    print(f"Average data loading time: {avg_batch_time:.3f}s per batch")

    if avg_batch_time > 1.0:
        print("⚠️  Data loading is slow - consider reducing num_workers or batch_size")

    # Test model forward pass
    print(f"\nTesting model...")
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
        
        if torch.isnan(loss):
            print("❌ Still getting NaN loss!")
            return
        else:
            print("✅ No NaN loss - ready to train!")

    # Training setup
    max_epochs = 10 if USE_TEST_DATASET else 5
    
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("tb_logs", name="patchtstnan_w3_improved")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=50,
        logger=logger,
        gradient_clip_val=1.0,
        # Performance optimizations
        enable_checkpointing=True,
        val_check_interval=0.25,
        limit_val_batches=50 if USE_TEST_DATASET else None,
        # Fast dev run for testing
        fast_dev_run=False
    )

    # Start training
    print(f"\nStarting training for {max_epochs} epochs...")
    print(f"Expected time per epoch: ~{len(train_loader) * avg_batch_time / 60:.1f} minutes")
    
    total_start_time = time.time()
    
    try:
        trainer.fit(model, data_module)
        
        total_time = time.time() - total_start_time
        print(f"\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Time per epoch: {total_time/max_epochs/60:.1f} minutes")
        
        if trainer.callback_metrics:
            print(f"\nFinal metrics:")
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.item():.4f}")
        
        print(f"\n✅ Success! No NaN loss issues.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()