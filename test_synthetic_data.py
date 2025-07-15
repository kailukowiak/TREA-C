import torch

from duet.data.dataset import SyntheticTimeSeriesDataset


# Create dataset
dataset = SyntheticTimeSeriesDataset(
    num_samples=100, T=64, C_num=4, C_cat=2, task="classification", num_classes=3
)

# Check correlation between mean of first channel and labels
first_channel_means = []
labels = []

for i in range(len(dataset)):
    sample = dataset[i]
    x_num = sample["x_num"]
    y = sample["y"]

    # Compute mean of first channel (ignoring NaNs)
    mean_val = torch.nanmean(x_num[0])
    first_channel_means.append(mean_val.item())
    labels.append(y.item())

# Convert to tensors for analysis
means = torch.tensor(first_channel_means)
labels = torch.tensor(labels)

# Group by label and show mean values
print("Mean of first channel grouped by label:")
for label in range(3):
    mask = labels == label
    if mask.any():
        label_means = means[mask]
        print(
            f"Label {label}: mean={label_means.mean():.3f}, "
            f"std={label_means.std():.3f}, count={mask.sum()}"
        )

# Show the deterministic mapping
print("\nSample mappings (first 20):")
sorted_indices = torch.argsort(means)[:20]
for idx in sorted_indices:
    print(f"Mean: {means[idx]:.3f} -> Label: {labels[idx]}")
