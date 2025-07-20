"""Simple DataModule that accepts pre-loaded datasets."""

import pytorch_lightning as pl

from torch.utils.data import DataLoader


class TimeSeriesDataModuleV2(pl.LightningDataModule):
    """Simple DataModule that accepts pre-loaded datasets."""

    def __init__(
        self,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_column_names(self) -> list[str] | None:
        """Get column names from the train dataset if available."""
        if hasattr(self.train_dataset, "get_column_names"):
            return self.train_dataset.get_column_names()
        elif hasattr(self.train_dataset, "COLUMN_NAMES"):
            return self.train_dataset.COLUMN_NAMES.copy()
        return None

    def get_feature_info(self) -> dict | None:
        """Get feature information from the train dataset if available."""
        if hasattr(self.train_dataset, "get_feature_info"):
            return self.train_dataset.get_feature_info()
        return None
