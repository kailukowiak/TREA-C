#!/usr/bin/env python3
"""
Multi-Dataset Training Demo: Comparison of column embedding strategies.

This script demonstrates how to train a model across multiple datasets
with different column embedding strategies for transferability.
"""

import os
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score


# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from duet.data.datamodule_v2 import TimeSeriesDataModuleV2
from duet.data.etth1 import ETTh1Dataset
from duet.models.multi_dataset_patch_duet import MultiDatasetPatchDuET
from duet.utils import get_checkpoint_path, get_output_path


def create_synthetic_dataset(
    num_samples: int = 1000,
    c_in: int = 5,
    seq_len: int = 96,
    num_classes: int = 3,
    column_names: list[str] = None,
    nan_rate: float = 0.05,
):
    """Create a synthetic dataset with specified characteristics."""

    if column_names is None:
        column_names = [f"sensor_{i}" for i in range(c_in)]

    # Generate synthetic data
    x_num = torch.randn(num_samples, c_in, seq_len)

    # Add some patterns to make classification meaningful
    for i in range(num_classes):
        start_idx = i * (num_samples // num_classes)
        end_idx = (i + 1) * (num_samples // num_classes)

        # Add class-specific patterns
        x_num[start_idx:end_idx, :, :] += (
            torch.sin(torch.linspace(0, 2 * np.pi * (i + 1), seq_len))
            .unsqueeze(0)
            .unsqueeze(0)
        )

    # Inject NaNs
    if nan_rate > 0:
        nan_mask = torch.rand_like(x_num) < nan_rate
        x_num[nan_mask] = float("nan")

    # Create labels
    y = torch.repeat_interleave(torch.arange(num_classes), num_samples // num_classes)

    # Handle remainder
    if len(y) < num_samples:
        remaining = num_samples - len(y)
        y = torch.cat([y, torch.randint(0, num_classes, (remaining,))])

    return {
        "x_num": x_num,
        "y": y,
        "column_names": column_names,
        "c_in": c_in,
        "seq_len": seq_len,
        "num_classes": num_classes,
    }


class SyntheticDataset(torch.utils.data.Dataset):
    """Wrapper for synthetic data."""

    def __init__(self, data_dict):
        self.x_num = data_dict["x_num"]
        self.y = data_dict["y"]
        self.column_names = data_dict["column_names"]
        self.c_in = data_dict["c_in"]
        self.sequence_length = data_dict["seq_len"]
        self.num_classes = data_dict["num_classes"]
        self.task = "classification"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x_num": self.x_num[idx], "y": self.y[idx]}

    def get_column_names(self):
        return self.column_names


def train_model(model, dm, model_name, max_epochs=10):
    """Train a model and return results."""

    # Callbacks
    checkpoint_dir = get_checkpoint_path(f"multi_dataset/{model_name}")
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        verbose=True,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True,
        logger=False,
        enable_progress_bar=False,  # Reduce output
    )

    # Train
    start_time = time.time()
    trainer.fit(model, dm)
    training_time = time.time() - start_time

    # Use the current model for evaluation (preserves column embedder state)
    # Loading from checkpoint can lose column embedder configuration
    best_model = model
    best_model.eval()

    # Get predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dm.val_dataloader():
            x_num = batch["x_num"]
            x_cat = batch.get("x_cat", None)
            labels = batch["y"]

            # Move to device
            device = next(best_model.parameters()).device
            x_num = x_num.to(device)
            if x_cat is not None:
                x_cat = x_cat.to(device)
            labels = labels.to(device)

            if labels.ndim == 2:
                labels = torch.argmax(labels, dim=1)
            labels = labels.long()

            logits = best_model(x_num, x_cat)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # Get final validation loss
    val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
    if torch.is_tensor(val_loss):
        val_loss = val_loss.item()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Val Loss": val_loss,
        "Training Time (s)": training_time,
        "Parameters": sum(p.numel() for p in model.parameters()),
        "Epochs": trainer.current_epoch + 1,
        "Embedding Stats": model.get_embedding_stats()
        if hasattr(model, "get_embedding_stats")
        else {},
    }


def multi_dataset_experiment():
    """Run multi-dataset training experiment."""

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("MULTI-DATASET TRAINING EXPERIMENT")
    print("=" * 80)

    # Create multiple synthetic datasets with different column patterns
    datasets = {
        "Temperature Sensors": create_synthetic_dataset(
            num_samples=800,
            c_in=4,
            column_names=["indoor_temp", "outdoor_temp", "humidity", "pressure"],
            nan_rate=0.05,
        ),
        "User Metrics": create_synthetic_dataset(
            num_samples=800,
            c_in=5,
            column_names=[
                "user_count",
                "session_duration",
                "page_views",
                "click_rate",
                "conversion_rate",
            ],
            nan_rate=0.03,
        ),
        "ETTh1": {
            "train": ETTh1Dataset(train=True),
            "val": ETTh1Dataset(train=False),
            "column_names": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "c_in": 7,
            "seq_len": 96,
            "num_classes": 3,
        },
    }

    # Test different strategies
    strategies = {
        "Baseline (No Columns)": "none",
        "Frozen BERT": "frozen_bert",
        "Auto-Expanding": "auto_expanding",
    }

    results = []

    for strategy_name, strategy in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"TESTING STRATEGY: {strategy_name}")
        print(f"{'=' * 60}")

        strategy_results = []

        for dataset_name, dataset_info in datasets.items():
            print(f"\nDataset: {dataset_name}")

            # Handle ETTh1 differently (real dataset)
            if dataset_name == "ETTh1":
                train_dataset = dataset_info["train"]
                val_dataset = dataset_info["val"]
                column_names = dataset_info["column_names"]
                c_in = dataset_info["c_in"]
                seq_len = dataset_info["seq_len"]
                num_classes = dataset_info["num_classes"]

                # Inject NaNs into ETTh1
                train_nan_mask = torch.rand_like(train_dataset.x_num) < 0.05
                val_nan_mask = torch.rand_like(val_dataset.x_num) < 0.05
                train_dataset.x_num[train_nan_mask] = float("nan")
                val_dataset.x_num[val_nan_mask] = float("nan")

            else:
                # Synthetic datasets
                train_data = dataset_info.copy()
                val_data = create_synthetic_dataset(
                    num_samples=200,  # Smaller validation set
                    c_in=train_data["c_in"],
                    seq_len=train_data["seq_len"],
                    num_classes=train_data["num_classes"],
                    column_names=train_data["column_names"],
                    nan_rate=0.05,
                )

                train_dataset = SyntheticDataset(train_data)
                val_dataset = SyntheticDataset(val_data)
                column_names = train_data["column_names"]
                c_in = train_data["c_in"]
                seq_len = train_data["seq_len"]
                num_classes = train_data["num_classes"]

            # Create data module
            dm = TimeSeriesDataModuleV2(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=32,
                num_workers=0,  # Avoid multiprocessing issues
            )

            # Create model
            if strategy == "none":
                model = MultiDatasetPatchDuET.create_baseline(
                    c_in=c_in,
                    seq_len=seq_len,
                    num_classes=num_classes,
                    d_model=64,  # Smaller for faster training
                    n_head=4,
                    num_layers=2,
                    lr=1e-3,
                )
            else:
                model = MultiDatasetPatchDuET.create_for_dataset(
                    c_in=c_in,
                    seq_len=seq_len,
                    num_classes=num_classes,
                    column_names=column_names,
                    strategy=strategy,
                    d_model=64,
                    n_head=4,
                    num_layers=2,
                    lr=1e-3,
                )

            print(f"  Columns: {column_names}")
            print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

            # Train model
            model_name = f"{strategy_name}_{dataset_name}".replace(" ", "_")
            result = train_model(model, dm, model_name, max_epochs=5)  # Faster training
            result["Dataset"] = dataset_name
            result["Strategy"] = strategy_name

            strategy_results.append(result)
            print(f"  Accuracy: {result['Accuracy']:.4f}")

        results.extend(strategy_results)

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("MULTI-DATASET EXPERIMENT RESULTS")
    print("=" * 80)

    # Summary by strategy
    summary = (
        df.groupby("Strategy")
        .agg(
            {
                "Accuracy": ["mean", "std"],
                "Parameters": "first",
                "Training Time (s)": "mean",
            }
        )
        .round(4)
    )

    print("\nSummary by Strategy:")
    print(summary)

    # Detailed results
    print("\nDetailed Results:")
    display_columns = [
        "Strategy",
        "Dataset",
        "Accuracy",
        "F1 Score",
        "Parameters",
        "Training Time (s)",
    ]
    print(df[display_columns].to_string(index=False, float_format="%.4f"))

    # Save results
    output_path = get_output_path("multi_dataset_experiment.csv", "experiments")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    baseline_acc = df[df["Strategy"] == "Baseline (No Columns)"]["Accuracy"].mean()
    bert_acc = df[df["Strategy"] == "Frozen BERT"]["Accuracy"].mean()
    auto_acc = df[df["Strategy"] == "Auto-Expanding"]["Accuracy"].mean()

    bert_improvement = (bert_acc - baseline_acc) / baseline_acc * 100
    auto_improvement = (auto_acc - baseline_acc) / baseline_acc * 100

    print("Average Accuracy Across Datasets:")
    print(f"  Baseline (No Columns):  {baseline_acc:.4f}")
    print(f"  Frozen BERT:           {bert_acc:.4f} ({bert_improvement:+.1f}%)")
    print(f"  Auto-Expanding:        {auto_acc:.4f} ({auto_improvement:+.1f}%)")

    # Parameter overhead
    baseline_params = df[df["Strategy"] == "Baseline (No Columns)"]["Parameters"].iloc[
        0
    ]
    bert_params = df[df["Strategy"] == "Frozen BERT"]["Parameters"].iloc[0]
    auto_params = df[df["Strategy"] == "Auto-Expanding"]["Parameters"].iloc[0]

    print("\nParameter Overhead:")
    print(f"  Baseline:       {baseline_params:,} params")
    print(
        f"  Frozen BERT:    {bert_params:,} params (+{bert_params - baseline_params:,})"
    )
    print(
        f"  Auto-Expanding: {auto_params:,} params (+{auto_params - baseline_params:,})"
    )

    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    if bert_acc > baseline_acc:
        bert_improvement = (bert_acc - baseline_acc) / baseline_acc * 100
        print(f"   ✅ Frozen BERT shows {bert_improvement:+.1f}% improvement!")
        print("      Recommended for multi-dataset training with diverse column names.")

    if auto_acc > baseline_acc:
        auto_improvement = (auto_acc - baseline_acc) / baseline_acc * 100
        print(f"   ✅ Auto-Expanding shows {auto_improvement:+.1f}% improvement!")
        print("      Good lightweight alternative to BERT.")

    print("   💡 For production: Use Frozen BERT for best transferability")
    print("      Pre-computed embeddings minimize inference overhead.")


if __name__ == "__main__":
    multi_dataset_experiment()
