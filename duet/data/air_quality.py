"""Air Quality dataset loader."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch.utils.data import Dataset


class AirQualityDataset(Dataset):
    """Air Quality monitoring dataset.

    This dataset contains air quality measurements from multiple sensors
    including pollutants, weather conditions, and location data.

    Features: 10 environmental sensors
    Classes: 3 air quality levels (Good, Moderate, Unhealthy)
    """

    COLUMN_NAMES = [
        "pm25",
        "pm10",
        "no2",
        "o3",
        "co",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "visibility",
    ]

    QUALITY_LABELS = ["good", "moderate", "unhealthy"]

    def __init__(
        self,
        data_dir: str = "./datasets/real_world/air_quality",
        split: str = "train",
        seq_len: int = 96,
        task: str = "classification",
        download: bool = True,
        nan_rate: float = 0.08,  # Higher NaN rate to simulate sensor failures
    ):
        """Initialize Air Quality dataset.

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
        """Download and prepare synthetic air quality data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        print("Creating synthetic air quality data...")

        np.random.seed(42)

        samples_per_class = 400 if self.split == "train" else 120

        data = []
        labels = []

        for quality_id, quality in enumerate(self.QUALITY_LABELS):
            for _ in range(samples_per_class):
                # Generate time-dependent patterns
                time_steps = np.linspace(0, 24, self.seq_len)  # 24 hours

                # Base pollution levels depend on air quality class
                if quality == "good":
                    base_pm25 = 10 + 5 * np.random.rand()
                    base_pm10 = 20 + 10 * np.random.rand()
                    base_no2 = 15 + 8 * np.random.rand()
                    base_o3 = 80 + 20 * np.random.rand()
                    base_co = 1 + 0.5 * np.random.rand()
                elif quality == "moderate":
                    base_pm25 = 35 + 15 * np.random.rand()
                    base_pm10 = 75 + 25 * np.random.rand()
                    base_no2 = 45 + 15 * np.random.rand()
                    base_o3 = 120 + 40 * np.random.rand()
                    base_co = 5 + 2 * np.random.rand()
                else:  # unhealthy
                    base_pm25 = 65 + 25 * np.random.rand()
                    base_pm10 = 150 + 50 * np.random.rand()
                    base_no2 = 85 + 25 * np.random.rand()
                    base_o3 = 200 + 50 * np.random.rand()
                    base_co = 12 + 4 * np.random.rand()

                # Add daily patterns (higher pollution during day)
                daily_pattern = 1 + 0.3 * np.sin(
                    2 * np.pi * time_steps / 24 - np.pi / 2
                )

                # Generate pollutant concentrations
                pm25 = base_pm25 * daily_pattern + 3 * np.random.randn(self.seq_len)
                pm10 = base_pm10 * daily_pattern + 5 * np.random.randn(self.seq_len)
                no2 = base_no2 * daily_pattern + 4 * np.random.randn(self.seq_len)
                o3 = base_o3 * (1.5 - 0.5 * daily_pattern) + 10 * np.random.randn(
                    self.seq_len
                )  # O3 anti-correlated
                co = base_co * daily_pattern + 0.5 * np.random.randn(self.seq_len)

                # Weather conditions (correlated with air quality)
                base_temp = 20 + 10 * np.sin(
                    2 * np.pi * time_steps / 24
                )  # Daily temp cycle
                base_humidity = (
                    60 - quality_id * 10
                )  # Lower humidity = worse air quality
                base_pressure = 1013 + 5 * np.sin(2 * np.pi * time_steps / 24)

                temperature = base_temp + 3 * np.random.randn(self.seq_len)
                humidity = (
                    base_humidity
                    + 0.3 * daily_pattern
                    + 5 * np.random.randn(self.seq_len)
                )
                pressure = base_pressure + 2 * np.random.randn(self.seq_len)
                wind_speed = (5 - quality_id * 1.5) + 2 * np.random.randn(
                    self.seq_len
                )  # Higher wind = better air
                visibility = (15 - quality_id * 4) + 3 * np.random.randn(
                    self.seq_len
                )  # Lower visibility = worse air

                # Ensure non-negative values for pollutants
                pm25 = np.maximum(pm25, 0)
                pm10 = np.maximum(pm10, 0)
                no2 = np.maximum(no2, 0)
                o3 = np.maximum(o3, 0)
                co = np.maximum(co, 0)
                wind_speed = np.maximum(wind_speed, 0)
                visibility = np.maximum(visibility, 1)

                # Stack features: [10, seq_len]
                sample = np.stack(
                    [
                        pm25,
                        pm10,
                        no2,
                        o3,
                        co,
                        temperature,
                        humidity,
                        pressure,
                        wind_speed,
                        visibility,
                    ]
                )

                data.append(sample)
                labels.append(quality_id)

        # Convert to numpy arrays
        data = np.array(data)  # [N, 10, seq_len]
        labels = np.array(labels)  # [N]

        # Save data
        np.save(self.data_dir / f"{self.split}_data.npy", data)
        np.save(self.data_dir / f"{self.split}_labels.npy", labels)

        print(f"Created {len(data)} samples with shape {data.shape}")

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load preprocessed data."""
        data_path = self.data_dir / f"{self.split}_data.npy"
        labels_path = self.data_dir / f"{self.split}_labels.npy"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data not found at {data_path}. Set download=True."
            )

        data = np.load(data_path)  # [N, 10, seq_len]
        labels = np.load(labels_path)  # [N]

        # Convert to tensors
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()

        # Inject random NaNs to simulate sensor failures
        if self.nan_rate > 0:
            nan_mask = torch.rand_like(data) < self.nan_rate
            data[nan_mask] = float("nan")

        return data, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"x_num": self.data[idx], "y": self.labels[idx]}

    def get_column_names(self) -> list[str]:
        return self.COLUMN_NAMES

    @property
    def num_classes(self) -> int:
        return len(self.QUALITY_LABELS)

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
    print("Testing Air Quality Dataset...")

    dataset = AirQualityDataset(split="train", seq_len=96, download=True)

    print(f"Dataset size: {len(dataset)}")
    print(
        f"Features: {dataset.numeric_features} numeric, "
        f"{dataset.categorical_features} categorical"
    )
    print(f"Classes: {dataset.num_classes}")
    print(f"Column names: {dataset.get_column_names()}")

    # Test sample
    sample = dataset[0]
    print(f"Sample shape: {sample['x_num'].shape}")
    print(f"Label: {sample['y']} ({dataset.QUALITY_LABELS[sample['y']]})")

    # Check for NaNs
    nan_count = torch.isnan(sample["x_num"]).sum()
    print(f"NaN count in sample: {nan_count}")

    print("✅ Air Quality Dataset ready!")
