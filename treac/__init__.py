"""TREA-C: Triple-Encoded Attention for Column-aware Time Series Analysis."""

__version__ = "0.1.0"

from treac.models import DualPatchTransformer, MultiDatasetModel, PatchTSTNan
from utils import SyntheticTimeSeriesDataset, TimeSeriesDataModule


__all__ = [
    "DualPatchTransformer",
    "MultiDatasetModel",
    "PatchTSTNan",
    "SyntheticTimeSeriesDataset",
    "TimeSeriesDataModule",
]
