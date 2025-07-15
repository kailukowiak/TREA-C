"""ETTh1 dataset loader for time series classification."""

import os
import urllib.request
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any


class ETTh1Dataset(Dataset):
    """ETTh1 dataset for time series classification.
    
    The ETTh1 dataset contains electricity transformer temperature data
    with 7 features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) measured hourly.
    
    For classification, we convert the continuous OT (Oil Temperature) values
    into 3 classes based on quantiles: Low, Medium, High.
    """

    def __init__(
        self,
        data_dir: str = "./data/etth1",
        train: bool = True,
        sequence_length: int = 96,
        task: str = "classification",
        num_classes: int = 3,
        download: bool = True,
    ):
        """Initialize ETTh1 dataset.
        
        Args:
            data_dir: Directory to store/load data
            train: If True, use training set; otherwise test set
            sequence_length: Length of input sequences
            task: 'classification' or 'regression' (only classification supported)
            num_classes: Number of classes for classification
            download: Whether to download data if not present
        """
        if task != "classification":
            raise ValueError("ETTh1Dataset currently only supports classification task")
            
        self.data_dir = data_dir
        self.train = train
        self.sequence_length = sequence_length
        self.task = task
        self.num_classes = num_classes
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Download data if needed
        if download:
            self._download_data()
            
        # Load and process data
        self._load_data()
        
    def _download_data(self):
        """Download ETTh1 dataset if not present."""
        file_path = os.path.join(self.data_dir, "ETTh1.csv")
        
        if not os.path.exists(file_path):
            print(f"Downloading ETTh1 dataset to {file_path}")
            url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
            urllib.request.urlretrieve(url, file_path)
            print("Download complete")
        else:
            print(f"ETTh1 dataset already exists at {file_path}")
            
    def _load_data(self):
        """Load and preprocess the ETTh1 dataset."""
        file_path = os.path.join(self.data_dir, "ETTh1.csv")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Remove date column and use numeric features
        numeric_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        data = df[numeric_cols].values
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            # Input sequence
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            # Target: classify the OT value at the end of sequence
            target_value = data[i + self.sequence_length, -1]  # OT is last column
            targets.append(target_value)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Convert targets to classification labels based on quantiles
        if self.num_classes == 3:
            # Low, Medium, High based on 33rd and 67th percentiles
            low_thresh = np.percentile(targets, 33.33)
            high_thresh = np.percentile(targets, 66.67)
            
            labels = np.zeros(len(targets), dtype=int)
            labels[targets >= high_thresh] = 2  # High
            labels[(targets >= low_thresh) & (targets < high_thresh)] = 1  # Medium
            labels[targets < low_thresh] = 0  # Low
        else:
            # Use quantiles for other numbers of classes
            quantiles = np.linspace(0, 100, self.num_classes + 1)
            thresholds = np.percentile(targets, quantiles[1:-1])
            
            labels = np.zeros(len(targets), dtype=int)
            for i, thresh in enumerate(thresholds):
                labels[targets >= thresh] = i + 1
        
        # Train/test split (80/20)
        split_idx = int(0.8 * len(sequences))
        
        if self.train:
            self.sequences = sequences[:split_idx]
            self.labels = labels[:split_idx]
        else:
            self.sequences = sequences[split_idx:]
            self.labels = labels[split_idx:]
        
        # Convert to tensors and transpose to [B, C, T] format
        self.x_num = torch.FloatTensor(self.sequences).transpose(1, 2)  # [B, C, T]
        self.y = torch.LongTensor(self.labels)
        
        # Create dummy categorical data (ETTh1 has no categorical features)
        self.x_cat = torch.zeros(len(self.sequences), 0, self.sequence_length, dtype=torch.long)
        
        print(f"Loaded ETTh1 dataset:")
        print(f"  Split: {'Train' if self.train else 'Test'}")
        print(f"  Samples: {len(self.sequences)}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Features: {self.x_num.shape[1]}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Class distribution: {np.bincount(self.labels)}")
        
    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature information for model configuration."""
        return {
            "n_numeric": self.x_num.shape[1],
            "n_categorical": 0,
            "cat_cardinalities": [],
            "sequence_length": self.sequence_length,
            "num_classes": self.num_classes,
            "task": self.task,
        }
        
    def __len__(self) -> int:
        return len(self.y)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_num": self.x_num[idx],
            "x_cat": self.x_cat[idx],
            "y": self.y[idx]
        }