import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, Dataset


class SyntheticTimeSeriesDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        T=64,
        C_num=4,
        C_cat=2,
        num_classes=3,
        task="classification",
    ):
        self.task = task
        self.num_classes = num_classes
        self.T = T
        self.C_num = C_num
        self.C_cat = C_cat

        self.x_num = torch.randn(num_samples, C_num, T)
        self.x_cat = torch.randint(0, 5, (num_samples, C_cat, T))

        if task == "classification":
            # Create a learnable pattern: class is based on the mean of the first
            # channel
            signal = self.x_num[:, 0, :].mean(dim=1)
            # Stretch the signal to make the pattern more pronounced
            stretched_signal = (signal - signal.mean()) / signal.std()
            # Quantize the signal to create class labels
            self.y = torch.quantize_per_tensor(
                stretched_signal, scale=1.0, zero_point=0, dtype=torch.qint8
            ).int_repr()
            # Clamp values to be within the number of classes
            self.y = torch.clamp(self.y, 0, num_classes - 1)
        else:  # regression
            # Create a learnable pattern: target is the mean of the first channel
            # + noise
            self.y = (
                self.x_num[:, 0, :].mean(dim=1, keepdim=True)
                + torch.randn(num_samples, 1) * 0.1
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x_num": self.x_num[idx], "x_cat": self.x_cat[idx], "y": self.y[idx]}


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, task="classification"):
        super().__init__()
        self.batch_size = batch_size
        self.task = task

    def setup(self, stage=None):
        self.train_dataset = SyntheticTimeSeriesDataset(task=self.task)
        self.val_dataset = SyntheticTimeSeriesDataset(task=self.task)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
