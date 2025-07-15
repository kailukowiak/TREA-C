import pandas as pd
import torch

from torch.utils.data import Dataset

from duet.data.config import DatasetConfig


class SyntheticTimeSeriesDataset(Dataset):
    """Synthetic dataset for testing and development."""

    def __init__(
        self,
        num_samples: int = 1000,
        T: int = 64,
        C_num: int = 4,
        C_cat: int = 2,
        cat_cardinalities: list | None = None,
        num_classes: int = 3,
        task: str = "classification",
        missing_ratio: float = 0.1,
    ):
        """Initialize synthetic dataset.

        Args:
            num_samples: Number of samples
            T: Sequence length
            C_num: Number of numeric features
            C_cat: Number of categorical features
            cat_cardinalities: List of cardinalities for each categorical feature
            num_classes: Number of classes for classification
            task: 'classification' or 'regression'
            missing_ratio: Ratio of missing values to introduce
        """
        self.task = task
        self.num_classes = num_classes
        self.T = T
        self.C_num = C_num
        self.C_cat = C_cat

        # Default cardinalities if not provided
        if cat_cardinalities is None:
            cat_cardinalities = [5] * C_cat
        self.cat_cardinalities = cat_cardinalities

        # Generate numeric data with some missing values
        self.x_num = torch.randn(num_samples, C_num, T)
        if missing_ratio > 0:
            mask = torch.rand(num_samples, C_num, T) < missing_ratio
            self.x_num[mask] = float("nan")

        # Generate categorical data
        self.x_cat = torch.stack(
            [
                torch.randint(0, cardinality, (num_samples, T))
                for cardinality in cat_cardinalities
            ],
            dim=1,
        )

        # Generate targets
        if task == "classification":
            # Create a learnable pattern: class is based on the mean of the first channel
            signal = self.x_num[:, 0, :].mean(dim=1)
            # Stretch the signal to make the pattern more pronounced
            stretched_signal = (signal - signal.mean()) / signal.std()
            # Quantize the signal to create class labels
            self.y = torch.quantize_per_tensor(stretched_signal, scale=1.0, zero_point=0, dtype=torch.qint8).int_repr()
            # Clamp values to be within the number of classes
            self.y = torch.clamp(self.y.long(), 0, num_classes - 1)
        else: # regression
            # Create a learnable pattern: target is the mean of the first channel + noise
            self.y = self.x_num[:, 0, :].mean(dim=1, keepdim=True) + torch.randn(num_samples, 1) * 0.1


    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x_num": self.x_num[idx], "x_cat": self.x_cat[idx], "y": self.y[idx]}


class TimeSeriesDataset(Dataset):
    """Dataset for real time series data from pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, config: DatasetConfig, transform=None):
        """Initialize dataset from DataFrame and config.

        Args:
            df: DataFrame with time series data
            config: DatasetConfig instance with metadata
            transform: Optional data transformations
        """
        self.config = config
        self.transform = transform

        # Process data based on config
        self._process_dataframe(df)

    def _process_dataframe(self, df: pd.DataFrame):
        """Process DataFrame into tensors."""
        # This assumes data is in a specific format
        # Actual implementation would depend on data structure

        n_samples = self.config.n_samples
        T = self.config.sequence_length

        # Initialize tensors
        self.x_num = torch.zeros(n_samples, self.config.n_numeric, T)
        self.x_cat = torch.zeros(
            n_samples, self.config.n_categorical, T, dtype=torch.long
        )

        # Process target
        if self.config.task == "classification":
            self.y = torch.zeros(n_samples, dtype=torch.long)
        else:
            self.y = torch.zeros(n_samples, 1)

        # TODO: Actual data processing logic would go here
        # This is a placeholder - real implementation would reshape
        # the DataFrame into the required tensor format

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {"x_num": self.x_num[idx], "x_cat": self.x_cat[idx], "y": self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample