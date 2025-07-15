"""DuET: Dual-Patch Transformer for Time Series with Categorical Features."""

__version__ = "0.1.0"

from duet.models import DualPatchTransformer
from duet.data import SyntheticTimeSeriesDataset, TimeSeriesDataModule

__all__ = [
    "DualPatchTransformer",
    "SyntheticTimeSeriesDataset", 
    "TimeSeriesDataModule",
]