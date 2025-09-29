"""DuET: Dual-Patch Transformer for Time Series with Categorical Features."""

__version__ = "0.1.0"

from utils import SyntheticTimeSeriesDataset, TimeSeriesDataModule
from duet.models import DualPatchTransformer


__all__ = [
    "DualPatchTransformer",
    "SyntheticTimeSeriesDataset",
    "TimeSeriesDataModule",
]
