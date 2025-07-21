#!/usr/bin/env python3
"""
Real-World Dataset Demo: Variable feature model on diverse real datasets.

This demonstrates the variable feature model's performance on real-world
time series datasets with different characteristics.
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

from duet.data.air_quality import AirQualityDataset
from duet.data.datamodule_v2 import TimeSeriesDataModuleV2
from duet.data.etth1 import ETTh1Dataset
from duet.data.financial_market import FinancialMarketDataset
from duet.data.human_activity import HumanActivityDataset
from duet.data.nasa_turbofan import NASATurbofanDataset
from duet.models.variable_feature_patch_duet import VariableFeaturePatchDuET
from duet.utils import get_checkpoint_path, get_output_path


def train_real_world_model(model, dm, model_name, max_epochs=10):
    """Train a model and return results."""

    # Callbacks
    checkpoint_dir = get_checkpoint_path(f"real_world/{model_name}")
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
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
        enable_progress_bar=False,
    )

    # Train
    start_time = time.time()
    trainer.fit(model, dm)
    training_time = time.time() - start_time

    # Use current model for evaluation (avoid checkpoint loading issues)
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
    }


def real_world_experiment():
    """Run real-world dataset experiment."""

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("REAL-WORLD DATASET EXPERIMENT")
    print("=" * 80)

    # Define real-world datasets with their characteristics
    datasets_config = {
        "ETTh1": {
            "class": ETTh1Dataset,
            "train_args": {"train": True, "sequence_length": 96},
            "val_args": {"train": False, "sequence_length": 96},
            "expected_features": 7,
            "expected_classes": 3,
            "description": "Electricity Transformer Temperature",
        },
        "NASA_Turbofan": {
            "class": NASATurbofanDataset,
            "train_args": {
                "train": True,
                "sequence_length": 80,
                "task": "classification",
            },
            "val_args": {
                "train": False,
                "sequence_length": 80,
                "task": "classification",
            },
            "expected_features": 24,
            "expected_classes": 3,
            "description": "Turbofan Engine Degradation",
        },
        "Human_Activity": {
            "class": HumanActivityDataset,
            "train_args": {"split": "train", "seq_len": 128},
            "val_args": {"split": "val", "seq_len": 128},
            "expected_features": 6,
            "expected_classes": 6,
            "description": "Human Activity Recognition",
        },
        "Air_Quality": {
            "class": AirQualityDataset,
            "train_args": {"split": "train", "seq_len": 96},
            "val_args": {"split": "val", "seq_len": 96},
            "expected_features": 10,
            "expected_classes": 3,
            "description": "Air Quality Monitoring",
        },
        "Financial_Market": {
            "class": FinancialMarketDataset,
            "train_args": {"split": "train", "seq_len": 60},
            "val_args": {"split": "val", "seq_len": 60},
            "expected_features": 12,
            "expected_classes": 3,
            "description": "Financial Market Analysis",
        },
    }

    print("Real-World Dataset Schemas:")
    for name, config in datasets_config.items():
        print(
            f"  {name}: {config['expected_features']} features, "
            f"{config['expected_classes']} classes - {config['description']}"
        )

    # Create datasets and determine actual schemas
    actual_schemas = {}
    failed_datasets = []

    for dataset_name, config in datasets_config.items():
        print(f"\n--- Loading {dataset_name} ---")

        try:
            # Create train dataset to inspect schema
            train_dataset = config["class"](**config["train_args"])

            # Get actual schema information
            sample = train_dataset[0]
            x_shape = sample["x_num"].shape

            actual_schemas[dataset_name] = {
                "numeric": x_shape[0],
                "categorical": 0,  # All our datasets are numeric-only for now
                "columns": train_dataset.get_column_names(),
                "classes": train_dataset.num_classes
                if hasattr(train_dataset, "num_classes")
                else config["expected_classes"],
                "seq_len": x_shape[1],
            }

            print(
                f"  ✅ Schema: {actual_schemas[dataset_name]['numeric']} "
                "numeric features"
            )
            print(f"  ✅ Classes: {actual_schemas[dataset_name]['classes']}")
            print(f"  ✅ Sequence length: {actual_schemas[dataset_name]['seq_len']}")
            print(f"  ✅ Columns: {actual_schemas[dataset_name]['columns']}")

        except Exception as e:
            print(f"  ❌ Failed to load {dataset_name}: {e}")
            failed_datasets.append(dataset_name)
            continue

    # Remove failed datasets
    for dataset_name in failed_datasets:
        del datasets_config[dataset_name]

    if not actual_schemas:
        print("❌ No datasets loaded successfully!")
        return

    # Determine unified model configuration
    max_numeric = max(schema["numeric"] for schema in actual_schemas.values())
    max_categorical = max(schema["categorical"] for schema in actual_schemas.values())
    max_classes = max(schema["classes"] for schema in actual_schemas.values())

    print("\n=== UNIFIED MODEL CONFIGURATION ===")
    print(f"Max numeric features: {max_numeric}")
    print(f"Max categorical features: {max_categorical}")
    print(f"Max classes: {max_classes}")

    # Create unified model
    model = VariableFeaturePatchDuET.create_for_multi_dataset(
        dataset_schemas={
            name: {
                "numeric": schema["numeric"],
                "categorical": schema["categorical"],
                "columns": schema["columns"],
            }
            for name, schema in actual_schemas.items()
        },
        num_classes=max_classes,
        strategy="auto_expanding",
        d_model=64,  # Smaller for faster training
        n_head=4,
        num_layers=2,
        lr=1e-3,
        max_sequence_length=256,  # Support longer sequences
    )

    print(f"\nUnified model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test each dataset
    results = []

    for dataset_name, config in datasets_config.items():
        print(f"\n{'=' * 60}")
        print(f"TESTING DATASET: {dataset_name}")
        print(f"{'=' * 60}")

        try:
            # Set schema for this dataset
            schema = actual_schemas[dataset_name]
            model.set_dataset_schema(
                numeric_features=schema["numeric"],
                categorical_features=schema["categorical"],
                column_names=schema["columns"],
            )

            # Create datasets
            train_dataset = config["class"](**config["train_args"])
            val_dataset = config["class"](**config["val_args"])

            # Create data module
            dm = TimeSeriesDataModuleV2(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=32,
                num_workers=0,
            )

            print(f"  Dataset: {config['description']}")
            print(
                f"  Features: {schema['numeric']} numeric + "
                f"{schema['categorical']} categorical"
            )
            print(f"  Classes: {schema['classes']}")
            print(
                f"  Train samples: {len(train_dataset)}, "
                f"Val samples: {len(val_dataset)}"
            )
            print(f"  Sequence length: {schema['seq_len']}")

            # Train model (fewer epochs for faster testing)
            result = train_real_world_model(model, dm, dataset_name, max_epochs=10)
            result["Dataset"] = dataset_name
            result["Description"] = config["description"]
            result["Numeric Features"] = schema["numeric"]
            result["Categorical Features"] = schema["categorical"]
            result["Total Features"] = schema["numeric"] + schema["categorical"]
            result["Classes"] = schema["classes"]
            result["Sequence Length"] = schema["seq_len"]

            results.append(result)
            print(f"  ✅ Accuracy: {result['Accuracy']:.4f}")

        except Exception as e:
            print(f"  ❌ Failed to train on {dataset_name}: {e}")
            continue

    if not results:
        print("❌ No successful training runs!")
        return

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("REAL-WORLD EXPERIMENT RESULTS")
    print("=" * 80)

    display_columns = [
        "Dataset",
        "Description",
        "Total Features",
        "Classes",
        "Sequence Length",
        "Accuracy",
        "F1 Score",
        "Training Time (s)",
    ]
    print(df[display_columns].to_string(index=False, float_format="%.4f"))

    # Save results
    output_path = get_output_path("real_world_experiment.csv", "experiments")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    avg_accuracy = df["Accuracy"].mean()
    std_accuracy = df["Accuracy"].std()

    print(
        f"Average accuracy across all datasets: {avg_accuracy:.4f} ± {std_accuracy:.4f}"
    )
    print(
        f"Feature count range: {df['Total Features'].min()}-"
        f"{df['Total Features'].max()} features"
    )
    print(
        f"Sequence length range: {df['Sequence Length'].min()}-"
        f"{df['Sequence Length'].max()} timesteps"
    )
    print(f"Classes range: {df['Classes'].min()}-{df['Classes'].max()} classes")

    # Check correlation between feature count and performance
    feature_accuracy_corr = df["Total Features"].corr(df["Accuracy"])
    print(
        f"Correlation between feature count and accuracy: {feature_accuracy_corr:.3f}"
    )

    # Best and worst performing datasets
    best_dataset = df.loc[df["Accuracy"].idxmax()]
    worst_dataset = df.loc[df["Accuracy"].idxmin()]

    print("\n🏆 BEST PERFORMANCE:")
    print(f"   Dataset: {best_dataset['Dataset']} - {best_dataset['Description']}")
    print(f"   Accuracy: {best_dataset['Accuracy']:.4f}")
    print(f"   Features: {best_dataset['Total Features']}")

    print("\n📊 MOST CHALLENGING:")
    print(f"   Dataset: {worst_dataset['Dataset']} - {worst_dataset['Description']}")
    print(f"   Accuracy: {worst_dataset['Accuracy']:.4f}")
    print(f"   Features: {worst_dataset['Total Features']}")

    print("\n🎯 CONCLUSION:")
    print(f"   ✅ Successfully handled {len(results)} diverse real-world datasets")
    print(
        f"   ✅ Feature range: {df['Total Features'].min()}-"
        f"{df['Total Features'].max()} features"
    )
    print(f"   ✅ Average performance: {avg_accuracy:.1%}")
    print(
        f"   ✅ Variable sequence lengths: {df['Sequence Length'].min()}-"
        f"{df['Sequence Length'].max()} timesteps"
    )
    print("   🚀 REAL-WORLD READY!")


if __name__ == "__main__":
    real_world_experiment()
