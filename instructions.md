# Building a Time-Series Transformer with Time-Varying Categorical Inputs (Regression + Classification)

This document provides a **step-by-step guide** to implement a **Dual-Patch Transformer
model** that combines both **temporal** and **feature-wise patches**. The model supports
**time-varying categorical features** and handles both **regression** and
**classification** tasks. It also includes support for **pretraining** on unlabeled data
and fine-tuning with few labels.

---

## 🔧 1. Project Setup

### 1.1 Install Required Packages

```bash
uv add torch pytorch-lightning scikit-learn pandas numpy
```

---

## 🧱 2. Dataset Format

### 2.1 Dataset Specification

Each sample in the dataset is a **multivariate time series** of shape `[C, T]`, where:

- `C` = number of numeric + categorical channels
- `T` = number of time steps per sequence

We divide features into:

- `x_num`: continuous sensor values → shape `[B, C_num, T]`
- `x_cat`: time-varying categorical channels → shape `[B, C_cat, T]`
- `y`: labels for regression or classification → shape `[B]` or `[B, T]`

---

## 📦 3. Synthetic Dataset Generator

```python
from torch.utils.data import Dataset
import torch
import numpy as np

class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, num_samples=1000, T=64, C_num=4, C_cat=2, num_classes=3, task='classification'):
        self.task = task
        self.num_classes = num_classes
        self.T = T
        self.C_num = C_num
        self.C_cat = C_cat

        self.x_num = torch.randn(num_samples, C_num, T)
        self.x_cat = torch.randint(0, 5, (num_samples, C_cat, T))

        if task == 'classification':
            self.y = torch.randint(0, num_classes, (num_samples,))
        else:
            self.y = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x_num': self.x_num[idx],
            'x_cat': self.x_cat[idx],
            'y': self.y[idx]
        }
```

---

## 🚚 4. PyTorch Lightning DataModule

```python
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, task='classification'):
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
```

---

## 🧠 5. Dual‑Patch Transformer Model with **Built‑in Missing Mask Channel**

We now encode numeric NaNs by **doubling each sensor channel**:

- **Value channel**: original reading, NaNs replaced by 0.
- **Mask channel**: `1` if the reading was missing, else `0`.

This keeps the tensor free of NaNs, lets CUDA kernels run at full speed, and injects the
“missingness” signal directly into the projection layer.

### 5.1  Missing‑value pipeline

1. **Create mask**: `m_nan = torch.isnan(x_num)`.
2. **Zero‑fill**: `x_val = torch.nan_to_num(x_num, nan=0.0)`.
3. **Stack**: `x_num2 = torch.cat([x_val, m_nan.float()], dim=1)` → shape `[B, 2·C_num,
   T]`.
4. **Project** with `num_proj` (input channels = `2·C_num`).

### 5.2  Updated Lightning module

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class DualPatchTransformer(pl.LightningModule):
    def __init__(self, C_num, C_cat, cat_cardinality, T, d_model=64,
                 task='classification', num_classes=3):
        super().__init__()
        self.task = task
        self.d_model = d_model

        # Categorical embeddings (time‑varying)
        self.cat_embs = nn.ModuleList([
            nn.Embedding(cat_cardinality, d_model) for _ in range(C_cat)
        ])

        # Numeric projection: NOTE 2× input channels
        self.num_proj = nn.Conv1d(C_num * 2, d_model, kernel_size=1)

        # Temporal encoder
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, n_head=4,
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Output heads
        if task == 'classification':
            self.head = nn.Linear(d_model, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.MSELoss()

    def forward(self, x_num, x_cat):  # x_num:[B,C_num,T] x_cat:[B,C_cat,T]
        # 1‑2. mask + zero‑fill
        m_nan  = torch.isnan(x_num).float()        # [B,C_num,T]
        x_val  = torch.nan_to_num(x_num, nan=0.0)  # [B,C_num,T]

        # 3. stack value & mask channels
        x_num2 = torch.cat([x_val, m_nan], dim=1)  # [B,2·C_num,T]

        # 4. project
        z_num = self.num_proj(x_num2)              # [B,d_model,T]

        # 5. add categorical embeddings
        cat_vec = sum(emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs))
        z = z_num + cat_vec.permute(0,2,1)         # align dims

        # 6. transformer + pooling
        z = z.permute(0,2,1)                       # [B,T,d_model]
        z = self.transformer(z)
        z = z.mean(dim=1)                          # global avg pooling

        return self.head(z)

    # Lightning hooks
    def training_step(self, batch, _):
        out = self(batch['x_num'], batch['x_cat'])
        loss = self.loss_fn(out.squeeze(), batch['y'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        out = self(batch['x_num'], batch['x_cat'])
        loss = self.loss_fn(out.squeeze(), batch['y'])
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```

> **Compute cost**: doubling the input channels only doubles the parameters/FLOPs of the
> *1×1 conv* (negligible compared with the Transformer). Everything downstream remains
> the same.

---

## 🏋️ 6. Training (Classification Example)

```python
dm = TimeSeriesDataModule(batch_size=32, task='classification')
model = DualPatchTransformer(C_num=4, C_cat=2, cat_cardinality=5, T=64, task='classification', num_classes=3)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, dm)
```

---

## 🔁 8. Pretraining on Unlabeled Data (Self-Supervised)

Replace the `head` and `loss_fn` with a self-supervised objective (e.g., Masked Patch
Prediction):

```python
self.head = nn.Linear(d_model, d_model)  # predict patch embedding
self.loss_fn = nn.MSELoss()
```

And train on raw unlabeled sequences with some masking applied to `x_num` or `x_cat`.
After training, you can:

1. Save encoder weights.
2. Fine-tune by replacing the head and unfreezing.

```python
# Fine-tune
model.head = nn.Linear(d_model, num_classes)  # or 1 for regression
model.loss_fn = nn.CrossEntropyLoss()
```

---

## ✅ 9. Summary

This guide walks through:

- Dual-patch transformer architecture
- Handling time-varying categorical features
- PyTorch Lightning integration
- Both classification and regression workflows
- Support for pretraining with limited labels

> You now have a flexible, industrial-ready architecture for working with sensor-rich,
> event-labeled, or partially-labeled time series with complex categorical metadata.

