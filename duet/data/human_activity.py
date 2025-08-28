"""Human Activity Recognition dataset loader."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch.utils.data import Dataset


class HumanActivityDataset(Dataset):
    """Human Activity Recognition dataset.

    This dataset contains accelerometer and gyroscope data from smartphones
    for recognizing human activities like walking, sitting, standing, etc.

    Features: 3 accelerometer + 3 gyroscope = 6 features
    Classes: 6 activities (walking, walking_upstairs, walking_downstairs, sitting,
    standing, laying)
    """

    COLUMN_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    ACTIVITY_LABELS = [
        "walking",
        "walking_upstairs",
        "walking_downstairs",
        "sitting",
        "standing",
        "laying",
    ]

    def __init__(
        self,
        data_dir: str = "./data/datasets/real_world/human_activity",
        split: str = "train",
        seq_len: int = 128,
        task: str = "classification",
        download: bool = True,
        nan_rate: float = 0.02,
    ):
        """Initialize Human Activity dataset.

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
        """Download and prepare synthetic human activity data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        print("Creating synthetic human activity data...")

        # Create synthetic accelerometer/gyroscope data for 6 activities
        np.random.seed(42)

        samples_per_activity = 500 if self.split == "train" else 150

        # Generate base signals for each activity type
        data = []
        labels = []

        for activity_id, activity in enumerate(self.ACTIVITY_LABELS):
            for _ in range(samples_per_activity):
                # Create activity-specific patterns
                time_steps = np.linspace(0, 4 * np.pi, self.seq_len)

                if "walking" in activity:
                    # Walking patterns: periodic acceleration with some randomness
                    freq = 2.0 + 0.5 * activity_id  # Different walking speeds
                    acc_x = np.sin(freq * time_steps) + 0.3 * np.random.randn(
                        self.seq_len
                    )
                    acc_y = np.cos(freq * time_steps) + 0.3 * np.random.randn(
                        self.seq_len
                    )
                    acc_z = (
                        9.8
                        + 0.5 * np.sin(0.5 * freq * time_steps)
                        + 0.2 * np.random.randn(self.seq_len)
                    )

                    # Gyroscope shows turning movements
                    gyro_x = 0.2 * np.sin(
                        1.5 * freq * time_steps
                    ) + 0.1 * np.random.randn(self.seq_len)
                    gyro_y = 0.2 * np.cos(
                        1.5 * freq * time_steps
                    ) + 0.1 * np.random.randn(self.seq_len)
                    gyro_z = 0.1 * np.sin(
                        0.8 * freq * time_steps
                    ) + 0.1 * np.random.randn(self.seq_len)

                elif activity == "sitting":
                    # Sitting: minimal movement, gravity dominates
                    acc_x = 0.1 * np.random.randn(self.seq_len)
                    acc_y = 0.1 * np.random.randn(self.seq_len)
                    acc_z = 9.8 + 0.1 * np.random.randn(self.seq_len)

                    gyro_x = 0.05 * np.random.randn(self.seq_len)
                    gyro_y = 0.05 * np.random.randn(self.seq_len)
                    gyro_z = 0.05 * np.random.randn(self.seq_len)

                elif activity == "standing":
                    # Standing: small postural adjustments
                    acc_x = 0.2 * np.random.randn(self.seq_len)
                    acc_y = 0.2 * np.random.randn(self.seq_len)
                    acc_z = 9.8 + 0.2 * np.random.randn(self.seq_len)

                    gyro_x = 0.1 * np.random.randn(self.seq_len)
                    gyro_y = 0.1 * np.random.randn(self.seq_len)
                    gyro_z = 0.1 * np.random.randn(self.seq_len)

                else:  # laying
                    # Laying: very minimal movement, different gravity orientation
                    acc_x = 9.8 + 0.1 * np.random.randn(
                        self.seq_len
                    )  # Gravity along x when laying
                    acc_y = 0.1 * np.random.randn(self.seq_len)
                    acc_z = 0.1 * np.random.randn(self.seq_len)

                    gyro_x = 0.03 * np.random.randn(self.seq_len)
                    gyro_y = 0.03 * np.random.randn(self.seq_len)
                    gyro_z = 0.03 * np.random.randn(self.seq_len)

                # Stack features: [6, seq_len]
                sample = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
                data.append(sample)
                labels.append(activity_id)

        # Convert to numpy arrays
        data = np.array(data)  # [N, 6, seq_len]
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

        data = np.load(data_path)  # [N, 6, seq_len]
        labels = np.load(labels_path)  # [N]

        # Convert to tensors
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()

        # Inject random NaNs for testing
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
        return len(self.ACTIVITY_LABELS)

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
    print("Testing Human Activity Dataset...")

    dataset = HumanActivityDataset(split="train", seq_len=128, download=True)

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
    print(f"Label: {sample['y']} ({dataset.ACTIVITY_LABELS[sample['y']]})")

    print("✅ Human Activity Dataset ready!")
