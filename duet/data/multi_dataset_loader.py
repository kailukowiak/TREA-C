"""Multi-dataset data loader for pre-training with variable schemas."""

import random
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl

from duet.utils.sequence_standardization import pad_or_truncate_sequence


def multi_dataset_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for multi-dataset batches."""
    result = {}
    
    # Handle each key separately
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key == "x_cat" and all(v is None for v in values):
            # All categorical features are None
            result[key] = None
        elif key in ["x_num", "feature_mask"]:
            # Stack tensors
            result[key] = torch.stack(values)
        elif key in ["y", "original_features", "original_seq_len", "num_classes"]:
            # Convert to tensor
            result[key] = torch.tensor(values)
        elif key == "dataset_name":
            # Keep as list of strings
            result[key] = values
        else:
            # Default behavior
            result[key] = values
    
    return result


class MultiDatasetSampler(Sampler):
    """Sampler that ensures balanced sampling across multiple datasets."""
    
    def __init__(
        self, 
        dataset_sizes: Dict[str, int], 
        batch_size: int,
        oversample_smaller: bool = True,
        seed: int = 42
    ):
        """Initialize multi-dataset sampler.
        
        Args:
            dataset_sizes: Dict mapping dataset names to their sizes
            batch_size: Batch size
            oversample_smaller: Whether to oversample smaller datasets
            seed: Random seed for reproducibility
        """
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.oversample_smaller = oversample_smaller
        self.seed = seed
        
        # Calculate dataset proportions
        total_samples = sum(dataset_sizes.values())
        self.dataset_weights = {
            name: size / total_samples 
            for name, size in dataset_sizes.items()
        }
        
        # Calculate samples per epoch
        if oversample_smaller:
            # Oversample to match largest dataset
            max_size = max(dataset_sizes.values())
            self.samples_per_epoch = max_size * len(dataset_sizes)
        else:
            self.samples_per_epoch = total_samples
    
    def __iter__(self):
        """Generate batch indices with balanced dataset sampling."""
        random.seed(self.seed)
        
        # Create indices for each dataset
        dataset_indices = {}
        cumulative_offset = 0
        
        for dataset_name, size in self.dataset_sizes.items():
            indices = list(range(cumulative_offset, cumulative_offset + size))
            
            if self.oversample_smaller:
                # Oversample to match largest dataset
                max_size = max(self.dataset_sizes.values())
                indices = indices * (max_size // size + 1)
                indices = indices[:max_size]
            
            random.shuffle(indices)
            dataset_indices[dataset_name] = indices
            cumulative_offset += size
        
        # Generate balanced batches
        dataset_names = list(self.dataset_sizes.keys())
        batch = []
        
        for _ in range(self.samples_per_epoch):
            # Choose dataset based on weights
            dataset_name = random.choices(
                dataset_names, 
                weights=[self.dataset_weights[name] for name in dataset_names]
            )[0]
            
            # Get next index from chosen dataset
            if dataset_indices[dataset_name]:
                batch.append(dataset_indices[dataset_name].pop(0))
            
            # Yield batch when full
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # Yield remaining samples
        if batch:
            yield batch
    
    def __len__(self):
        return self.samples_per_epoch // self.batch_size


class MultiDatasetCollection(Dataset):
    """Collection of multiple datasets with variable schemas."""
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        target_sequence_length: int = 96,
        pad_value: float = float('nan'),
        truncate_strategy: str = "random",
        max_numeric_features: int = None,
        max_categorical_features: int = None
    ):
        """Initialize multi-dataset collection.
        
        Args:
            datasets: Dict mapping dataset names to dataset objects
            target_sequence_length: Target sequence length for standardization
            pad_value: Value to use for padding sequences
            truncate_strategy: Strategy for truncating long sequences
        """
        self.datasets = datasets
        self.target_sequence_length = target_sequence_length
        self.pad_value = pad_value
        self.truncate_strategy = truncate_strategy
        
        # Build dataset metadata
        self.dataset_info = {}
        self.dataset_offsets = {}
        self.total_samples = 0
        
        for dataset_name, dataset in datasets.items():
            # Get dataset schema
            sample = dataset[0]
            x_shape = sample["x_num"].shape
            
            info = {
                "name": dataset_name,
                "size": len(dataset),
                "numeric_features": x_shape[0],
                "categorical_features": 0,  # Assume numeric-only for now
                "sequence_length": x_shape[1],
                "num_classes": getattr(dataset, 'num_classes', 3),
                "column_names": dataset.get_column_names() if hasattr(dataset, 'get_column_names') else []
            }
            
            self.dataset_info[dataset_name] = info
            self.dataset_offsets[dataset_name] = self.total_samples
            self.total_samples += len(dataset)
        
        # Calculate unified schema
        if max_numeric_features is not None:
            self.max_numeric_features = max_numeric_features
        else:
            self.max_numeric_features = max(info["numeric_features"] for info in self.dataset_info.values())
            
        if max_categorical_features is not None:
            self.max_categorical_features = max_categorical_features
        else:
            self.max_categorical_features = max(info["categorical_features"] for info in self.dataset_info.values())
            
        self.max_classes = max(info["num_classes"] for info in self.dataset_info.values())
        
        print(f"Multi-dataset collection initialized:")
        print(f"  Datasets: {len(datasets)}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Max numeric features: {self.max_numeric_features}")
        print(f"  Max classes: {self.max_classes}")
        print(f"  Target sequence length: {target_sequence_length}")
    
    def get_dataset_and_index(self, global_idx: int) -> Tuple[str, int]:
        """Convert global index to dataset name and local index."""
        for dataset_name, offset in self.dataset_offsets.items():
            dataset_size = self.dataset_info[dataset_name]["size"]
            if offset <= global_idx < offset + dataset_size:
                return dataset_name, global_idx - offset
        
        raise IndexError(f"Global index {global_idx} out of range")
    
    def pad_features_to_unified_space(
        self, 
        x_num: torch.Tensor, 
        dataset_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad features to unified space.
        
        Args:
            x_num: Numeric features [C, T]
            dataset_info: Information about the source dataset
            
        Returns:
            Tuple of (padded_features, feature_mask)
        """
        C, T = x_num.shape
        
        # Pad features to max size
        if C < self.max_numeric_features:
            padding = torch.full(
                (self.max_numeric_features - C, T), 
                self.pad_value,
                dtype=x_num.dtype,
                device=x_num.device
            )
            x_padded = torch.cat([x_num, padding], dim=0)
        else:
            x_padded = x_num
        
        # Create feature mask (1 = real feature, 0 = padding)
        feature_mask = torch.zeros(self.max_numeric_features, T)
        feature_mask[:C, :] = 1.0
        
        return x_padded, feature_mask
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with unified schema."""
        # Get dataset and local index
        dataset_name, local_idx = self.get_dataset_and_index(idx)
        dataset = self.datasets[dataset_name]
        dataset_info = self.dataset_info[dataset_name]
        
        # Get original sample
        sample = dataset[local_idx]
        x_num = sample["x_num"]  # [C, T]
        y = sample["y"]
        
        # Standardize sequence length
        x_num = pad_or_truncate_sequence(
            x_num.unsqueeze(0),  # Add batch dim
            self.target_sequence_length,
            pad_value=self.pad_value,
            truncate_strategy=self.truncate_strategy
        ).squeeze(0)  # Remove batch dim
        
        # Pad features to unified space
        x_padded, feature_mask = self.pad_features_to_unified_space(x_num, dataset_info)
        
        return {
            "x_num": x_padded,  # [max_features, target_seq_len]
            "x_cat": None,  # No categorical features for now
            "y": y,
            "feature_mask": feature_mask,  # [max_features, target_seq_len]
            "dataset_name": dataset_name,
            "original_features": dataset_info["numeric_features"],
            "original_seq_len": dataset_info["sequence_length"],
            "num_classes": dataset_info["num_classes"]
        }


class MultiDatasetDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for multi-dataset pre-training."""
    
    def __init__(
        self,
        train_datasets: Dict[str, Dataset],
        val_datasets: Dict[str, Dataset],
        batch_size: int = 32,
        num_workers: int = 0,
        target_sequence_length: int = 96,
        oversample_smaller: bool = True,
        seed: int = 42
    ):
        """Initialize multi-dataset data module.
        
        Args:
            train_datasets: Dict of training datasets
            val_datasets: Dict of validation datasets
            batch_size: Batch size
            num_workers: Number of data loading workers
            target_sequence_length: Target sequence length
            oversample_smaller: Whether to oversample smaller datasets
            seed: Random seed
        """
        super().__init__()
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_sequence_length = target_sequence_length
        self.oversample_smaller = oversample_smaller
        self.seed = seed
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            self.train_collection = MultiDatasetCollection(
                self.train_datasets,
                target_sequence_length=self.target_sequence_length
            )
            
            self.val_collection = MultiDatasetCollection(
                self.val_datasets,
                target_sequence_length=self.target_sequence_length
            )
    
    def train_dataloader(self):
        """Create training data loader with balanced sampling."""
        dataset_sizes = {
            name: len(dataset) 
            for name, dataset in self.train_datasets.items()
        }
        
        sampler = MultiDatasetSampler(
            dataset_sizes=dataset_sizes,
            batch_size=self.batch_size,
            oversample_smaller=self.oversample_smaller,
            seed=self.seed
        )
        
        return DataLoader(
            self.train_collection,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=multi_dataset_collate_fn
        )
    
    def val_dataloader(self):
        """Create validation data loaders (separate for each dataset)."""
        val_loaders = {}
        
        for dataset_name, dataset in self.val_datasets.items():
            # Create single-dataset collection for evaluation
            # Use the same max features as the training collection
            single_collection = MultiDatasetCollection(
                {dataset_name: dataset},
                target_sequence_length=self.target_sequence_length,
                max_numeric_features=self.train_collection.max_numeric_features,
                max_categorical_features=self.train_collection.max_categorical_features
            )
            
            val_loaders[dataset_name] = DataLoader(
                single_collection,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=multi_dataset_collate_fn
            )
        
        return val_loaders


if __name__ == "__main__":
    # Test the multi-dataset loader
    print("Testing MultiDatasetLoader...")
    
    from duet.data.etth1 import ETTh1Dataset
    from duet.data.human_activity import HumanActivityDataset
    from duet.data.air_quality import AirQualityDataset
    
    # Create test datasets
    train_datasets = {
        "etth1": ETTh1Dataset(train=True, sequence_length=96),
        "human_activity": HumanActivityDataset(split="train", seq_len=128),
        "air_quality": AirQualityDataset(split="train", seq_len=96)
    }
    
    val_datasets = {
        "etth1": ETTh1Dataset(train=False, sequence_length=96),
        "human_activity": HumanActivityDataset(split="val", seq_len=128),
        "air_quality": AirQualityDataset(split="val", seq_len=96)
    }
    
    # Create data module
    dm = MultiDatasetDataModule(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        batch_size=16,
        target_sequence_length=96
    )
    
    dm.setup()
    
    # Test training loader
    train_loader = dm.train_dataloader()
    print(f"Training batches: {len(train_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"x_num shape: {batch['x_num'].shape}")
    print(f"feature_mask shape: {batch['feature_mask'].shape}")
    print(f"Dataset names: {batch['dataset_name']}")
    print(f"Original features: {batch['original_features']}")
    
    # Test validation loaders
    val_loaders = dm.val_dataloader()
    print(f"Validation datasets: {list(val_loaders.keys())}")
    
    for dataset_name, loader in val_loaders.items():
        batch = next(iter(loader))
        print(f"{dataset_name} - x_num: {batch['x_num'].shape}, dataset: {batch['dataset_name'][0]}")
    
    print("✅ MultiDatasetLoader test complete!")