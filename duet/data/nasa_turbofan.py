"""NASA Turbofan Engine Degradation Dataset (C-MAPSS) loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class NASATurbofanDataset(Dataset):
    """NASA C-MAPSS Turbofan Engine Degradation Dataset.

    This dataset contains run-to-failure simulated data from turbofan engines.
    Each engine starts with different degrees of initial wear and manufacturing
    variation.

    Features:
    - 21 sensor channels
    - 3 operational settings
    - Multiple engines with varying lifespans
    - Task: Predict Remaining Useful Life (RUL) or classify failure risk
    """

    # Column names for the dataset
    index_names = ["unit_nr", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = [f"s_{i + 1}" for i in range(21)]
    col_names = index_names + setting_names + sensor_names

    def __init__(
        self,
        data_dir: str = "./data/nasa_turbofan",
        subset: str = "FD001",  # FD001, FD002, FD003, or FD004
        train: bool = True,
        sequence_length: int = 50,
        task: str = "classification",  # or "regression" for RUL
        num_classes: int = 3,  # for classification: healthy, degrading, critical
        download: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.train = train
        self.sequence_length = sequence_length
        self.task = task
        self.num_classes = num_classes

        if download and not self._check_exists():
            self._download()

        # Load data
        self.data, self.labels = self._load_data()

        # Preprocess sequences
        self.sequences, self.targets = self._create_sequences()

    def _check_exists(self):
        """Check if dataset already exists."""
        train_file = self.data_dir / f"train_{self.subset}.txt"
        test_file = self.data_dir / f"test_{self.subset}.txt"
        return train_file.exists() and test_file.exists()

    def _download(self):
        """Download NASA C-MAPSS dataset using kagglehub."""
        import shutil

        import kagglehub

        print("Downloading NASA C-MAPSS dataset via kagglehub...")

        # Download dataset
        path = kagglehub.dataset_download("behrad3d/nasa-cmaps")

        # Copy files to our data directory
        source_path = Path(path) / "CMaps"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        for file in source_path.glob("*.txt"):
            shutil.copy2(file, self.data_dir / file.name)

        print(f"Dataset downloaded to: {self.data_dir}")

    def _load_data(self):
        """Load and parse the turbofan data."""
        # Load train or test data
        filename = f"{'train' if self.train else 'test'}_{self.subset}.txt"
        filepath = self.data_dir / filename

        # Read data
        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=self.col_names)

        # Load RUL values for test set
        if not self.train:
            rul_file = self.data_dir / f"RUL_{self.subset}.txt"
            rul = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL"])

        # Group by unit
        grouped = df.groupby("unit_nr")

        data_list = []
        labels_list = []

        for unit_nr, unit_df in grouped:
            # Extract sensor data and settings
            sensors = unit_df[self.sensor_names].values
            settings = unit_df[self.setting_names].values

            # Combine sensors and settings
            features = np.hstack([sensors, settings])

            # Calculate RUL for each time step
            max_cycle = len(unit_df)
            if self.train:
                # For training, RUL = max_cycle - current_cycle
                rul = np.arange(max_cycle, 0, -1)
            else:
                # For test, use provided RUL for last timestep
                rul_last = rul.iloc[unit_nr - 1, 0]
                rul = np.arange(max_cycle + rul_last, rul_last, -1)

            data_list.append(features)
            labels_list.append(rul)

        return data_list, labels_list

    def _create_sequences(self):
        """Create fixed-length sequences from variable-length engine runs."""
        sequences = []
        targets = []

        for features, rul in zip(self.data, self.labels, strict=False):
            # Create sliding windows
            for i in range(len(features) - self.sequence_length + 1):
                seq = features[i : i + self.sequence_length]

                # Get target for end of sequence
                target_rul = rul[i + self.sequence_length - 1]

                if self.task == "classification":
                    # Convert RUL to classes
                    if target_rul > 125:
                        target = 0  # Healthy
                    elif target_rul > 50:
                        target = 1  # Degrading
                    else:
                        target = 2  # Critical
                else:
                    # Regression: predict RUL directly
                    target = target_rul

                sequences.append(seq)
                targets.append(target)

        # Convert to tensors
        sequences = torch.FloatTensor(sequences).transpose(1, 2)  # [B, C, T]
        if self.task == "classification":
            targets = torch.LongTensor(targets)
        else:
            targets = torch.FloatTensor(targets).unsqueeze(1)

        return sequences, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # NASA data has no categorical features, so create dummy ones
        x_num = self.sequences[idx][:21]  # 21 sensors
        x_cat = torch.zeros(2, self.sequence_length, dtype=torch.long)  # dummy

        return {"x_num": x_num, "x_cat": x_cat, "y": self.targets[idx]}

    def get_feature_info(self):
        """Get information about features."""
        return {
            "n_numeric": 21,  # 21 sensors
            "n_categorical": 2,  # dummy for compatibility
            "cat_cardinalities": [1, 1],  # dummy
            "sensor_names": self.sensor_names,
            "setting_names": self.setting_names,
        }
