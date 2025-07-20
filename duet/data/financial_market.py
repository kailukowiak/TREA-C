"""Financial Market dataset loader."""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FinancialMarketDataset(Dataset):
    """Financial Market time series dataset.
    
    This dataset contains simulated financial market data including
    price movements, technical indicators, and volume data.
    
    Features: 12 financial indicators
    Classes: 3 market directions (Bear, Neutral, Bull)
    """
    
    COLUMN_NAMES = [
        "open", "high", "low", "close", "volume", "volatility",
        "rsi", "macd", "bollinger_upper", "bollinger_lower", 
        "moving_avg_20", "moving_avg_50"
    ]
    
    MARKET_LABELS = ["bear", "neutral", "bull"]
    
    def __init__(
        self,
        data_dir: str = "./datasets/real_world/financial_market",
        split: str = "train",
        seq_len: int = 60,  # 60 time steps (e.g., minutes/hours)
        task: str = "classification",
        download: bool = True,
        nan_rate: float = 0.03,
    ):
        """Initialize Financial Market dataset.
        
        Args:
            data_dir: Directory to store/load data
            split: 'train', 'val', or 'test'
            seq_len: Sequence length for time series windows
            task: 'classification' or 'regression'
            download: Whether to download data if not found
            nan_rate: Rate of random NaN injection for testing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.task = task
        self.nan_rate = nan_rate
        
        if download and not (self.data_dir / f"{self.split}_data.npy").exists():
            self._download_and_prepare()
        
        self.data, self.labels = self._load_data()
        
    def _download_and_prepare(self):
        """Download and prepare synthetic financial market data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print("Creating synthetic financial market data...")
        
        np.random.seed(42)
        
        samples_per_class = 600 if self.split == "train" else 180
        total_samples = len(self.MARKET_LABELS) * samples_per_class
        
        data = []
        labels = []
        
        for market_id, market_type in enumerate(self.MARKET_LABELS):
            for _ in range(samples_per_class):
                # Generate price movement patterns based on market type
                if market_type == "bear":
                    # Downward trend with high volatility
                    trend = -0.02
                    volatility_base = 0.03
                elif market_type == "neutral":
                    # Sideways movement with moderate volatility
                    trend = 0.001
                    volatility_base = 0.015
                else:  # bull
                    # Upward trend with moderate volatility
                    trend = 0.015
                    volatility_base = 0.02
                
                # Generate price series using geometric Brownian motion
                dt = 1/252  # Daily time step
                price_changes = np.random.normal(trend * dt, volatility_base * np.sqrt(dt), self.seq_len)
                log_prices = np.cumsum(price_changes)
                prices = 100 * np.exp(log_prices)  # Start at $100
                
                # OHLC data
                open_prices = prices
                close_prices = np.roll(prices, -1)  # Next period's open becomes current close
                close_prices[-1] = prices[-1] * (1 + np.random.normal(trend * dt, volatility_base * np.sqrt(dt)))
                
                # High and low with realistic spreads
                high_prices = np.maximum(open_prices, close_prices) + np.random.exponential(0.5, self.seq_len)
                low_prices = np.minimum(open_prices, close_prices) - np.random.exponential(0.5, self.seq_len)
                
                # Volume (anti-correlated with price in bear markets)
                base_volume = 1000000
                if market_type == "bear":
                    volume = base_volume * (2 - (prices / prices[0])) + 200000 * np.random.rand(self.seq_len)
                else:
                    volume = base_volume * (1 + 0.5 * np.random.rand(self.seq_len))
                
                # Volatility (realized)
                volatility = np.abs(np.diff(np.log(prices), prepend=np.log(prices[0])))
                volatility = np.convolve(volatility, np.ones(5)/5, mode='same')  # 5-period smoothing
                
                # Technical indicators
                # RSI (Relative Strength Index)
                rsi = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, self.seq_len)) + 10 * np.random.randn(self.seq_len)
                rsi = np.clip(rsi, 0, 100)
                
                # MACD (Moving Average Convergence Divergence)
                macd = np.gradient(prices) + 0.5 * np.random.randn(self.seq_len)
                
                # Bollinger Bands
                moving_avg_20 = np.convolve(close_prices, np.ones(min(20, self.seq_len))/min(20, self.seq_len), mode='same')
                std_20 = np.std(close_prices) * np.ones(self.seq_len)  # Simplified
                bollinger_upper = moving_avg_20 + 2 * std_20
                bollinger_lower = moving_avg_20 - 2 * std_20
                
                # Moving averages
                moving_avg_50 = np.convolve(close_prices, np.ones(min(50, self.seq_len))/min(50, self.seq_len), mode='same')
                
                # Stack features: [12, seq_len]
                sample = np.stack([
                    open_prices, high_prices, low_prices, close_prices, volume, volatility,
                    rsi, macd, bollinger_upper, bollinger_lower, moving_avg_20, moving_avg_50
                ])
                
                data.append(sample)
                labels.append(market_id)
        
        # Convert to numpy arrays
        data = np.array(data)  # [N, 12, seq_len]
        labels = np.array(labels)  # [N]
        
        # Save data
        np.save(self.data_dir / f"{self.split}_data.npy", data)
        np.save(self.data_dir / f"{self.split}_labels.npy", labels)
        
        print(f"Created {len(data)} samples with shape {data.shape}")
    
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load preprocessed data."""
        data_path = self.data_dir / f"{self.split}_data.npy"
        labels_path = self.data_dir / f"{self.split}_labels.npy"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found at {data_path}. Set download=True.")
        
        data = np.load(data_path)  # [N, 12, seq_len]
        labels = np.load(labels_path)  # [N]
        
        # Convert to tensors
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        
        # Inject random NaNs to simulate missing data
        if self.nan_rate > 0:
            nan_mask = torch.rand_like(data) < self.nan_rate
            data[nan_mask] = float('nan')
        
        return data, labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "x_num": self.data[idx],
            "y": self.labels[idx]
        }
    
    def get_column_names(self) -> list[str]:
        return self.COLUMN_NAMES
    
    @property
    def num_classes(self) -> int:
        return len(self.MARKET_LABELS)
    
    @property
    def numeric_features(self) -> int:
        return len(self.COLUMN_NAMES)
    
    @property
    def categorical_features(self) -> int:
        return 0  # No categorical features
    
    @property
    def sequence_length(self) -> int:
        return self.seq_len


if __name__ == "__main__":
    # Test the dataset
    print("Testing Financial Market Dataset...")
    
    dataset = FinancialMarketDataset(
        split="train",
        seq_len=60,
        download=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Features: {dataset.numeric_features} numeric, {dataset.categorical_features} categorical")
    print(f"Classes: {dataset.num_classes}")
    print(f"Column names: {dataset.get_column_names()}")
    
    # Test sample
    sample = dataset[0]
    print(f"Sample shape: {sample['x_num'].shape}")
    print(f"Label: {sample['y']} ({dataset.MARKET_LABELS[sample['y']]})")
    
    print("✅ Financial Market Dataset ready!")