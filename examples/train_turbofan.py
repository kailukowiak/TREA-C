"""Train DualPatchTransformer on NASA Turbofan dataset."""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from duet.data.nasa_turbofan import NASATurbofanDataset
from duet.data.datamodule_v2 import TimeSeriesDataModuleV2
from duet.models.transformer import DualPatchTransformer


def main():
    # Try to load NASA dataset
    try:
        # First check if data exists
        train_dataset = NASATurbofanDataset(
            data_dir="./data/nasa_turbofan",
            subset="FD001",
            train=True,
            sequence_length=50,
            task="classification",
            num_classes=3,
            download=True  # Auto-download via kagglehub
        )
        
        val_dataset = NASATurbofanDataset(
            data_dir="./data/nasa_turbofan",
            subset="FD001",
            train=False,
            sequence_length=50,
            task="classification",
            num_classes=3,
            download=False
        )
        
        print(f"Loaded NASA Turbofan dataset:")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    except Exception as e:
        print("\n" + "="*60)
        print("NASA Turbofan dataset not found!")
        print("="*60)
        print("\nTo use this dataset, please:")
        print("1. Download from: https://data.nasa.gov/download/nk8v-ckry/application%2Fzip")
        print("2. Extract the zip file")
        print("3. Copy the .txt files to: ./data/nasa_turbofan/")
        print("\nAlternatively, let's use a publicly available dataset...")
        
        # Fallback to a different dataset or provide instructions
        print("\n" + "="*60)
        print("Alternative: Using Synthetic Data with Realistic Parameters")
        print("="*60)
        
        from duet.data.dataset import SyntheticTimeSeriesDataset
        
        # Create synthetic data mimicking industrial sensors
        train_dataset = SyntheticTimeSeriesDataset(
            num_samples=10000,
            T=100,  # Longer sequences
            C_num=21,  # Similar to turbofan sensors
            C_cat=3,   # Operational modes
            cat_cardinalities=[5, 10, 3],  # Different categorical ranges
            num_classes=3,
            task="classification",
            missing_ratio=0.05  # Some missing data
        )
        
        val_dataset = SyntheticTimeSeriesDataset(
            num_samples=2000,
            T=100,
            C_num=21,
            C_cat=3,
            cat_cardinalities=[5, 10, 3],
            num_classes=3,
            task="classification",
            missing_ratio=0.05
        )
        
        print("\nUsing synthetic data mimicking industrial sensors:")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    # Create data module
    dm = TimeSeriesDataModuleV2(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64,
        num_workers=4
    )
    
    # Get feature info
    if hasattr(train_dataset, 'get_feature_info'):
        feature_info = train_dataset.get_feature_info()
    else:
        feature_info = {
            'n_numeric': train_dataset.C_num,
            'n_categorical': train_dataset.C_cat,
            'cat_cardinalities': train_dataset.cat_cardinalities
        }
    
    # Create model
    model = DualPatchTransformer(
        C_num=feature_info['n_numeric'],
        C_cat=feature_info['n_categorical'],
        cat_cardinalities=feature_info['cat_cardinalities'],
        T=train_dataset[0]['x_num'].shape[1],  # sequence length
        d_model=128,  # Larger model for complex data
        nhead=8,
        num_layers=4,
        task="classification",
        num_classes=3,
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/turbofan',
        filename='duet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Logger
    logger = TensorBoardLogger('logs', name='nasa_turbofan_classification')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, dm)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()