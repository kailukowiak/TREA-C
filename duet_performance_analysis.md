# DuET Model Performance Analysis and Improvement Suggestions

## Current Performance Summary

| Model | Accuracy | F1 Score | Val Loss | Parameters | Training Time |
|-------|----------|----------|----------|------------|---------------|
| DuET | 74.4% | 0.741 | 0.813 | 844,611 | 35.2s |
| CNN | 93.1% | 0.932 | 0.220 | 684,931 | 22.7s |
| HF-PatchTST | 50.3% | 0.552 | 1.657 | 252,291 | 35.8s |

## Key Issues Identified

### 1. **Architecture Mismatch for Time Series Classification**
- **Problem**: DuET uses dual-patch (value+mask) encoding designed for handling missing values, but ETTh1 has no missing values
- **Impact**: Doubles input dimension (7→14 channels) without benefit, increasing parameters and complexity
- **Evidence**: x_num2 shape shows [B, 14, 96] after dual-patch encoding

### 2. **Suboptimal Hyperparameters**
- **d_model=64**: Too small for the complexity of the task
- **num_layers=3**: May be too shallow for learning complex temporal patterns
- **Learning rate=1e-3**: Standard but may not be optimal for this dataset
- **No learning rate scheduling**: Could benefit from warm-up and decay

### 3. **Data Preprocessing Issues**
- **No normalization**: Data ranges from 0.355 to 33.133 with std=7.65
- **Imbalanced classes**: [3900, 4216, 5743] distribution
- **Classification scheme**: Quantile-based binning may not be optimal

### 4. **Model Configuration Issues**
- **Pooling strategy**: Using mean pooling, but "cls" or "last" might be better for classification
- **Sequence length**: 96 might be too long for the temporal patterns in ETTh1
- **Attention heads**: 4 heads may be insufficient for d_model=64

## Specific Improvement Suggestions

### 1. **Architecture Fixes**

#### Option A: Conditional Dual-Patch Encoding
```python
# In DualPatchTransformer.__init__()
self.use_dual_patch = torch.isnan(x_num).any()  # Check if any NaNs exist

# In forward()
if self.use_dual_patch:
    m_nan = torch.isnan(x_num).float()
    x_val = torch.nan_to_num(x_num, nan=0.0)
    x_num2 = torch.cat([x_val, m_nan], dim=1)
else:
    x_num2 = x_num  # Direct projection without dual-patch
```

#### Option B: Add Input Normalization Layer
```python
# In __init__()
self.input_norm = nn.LayerNorm(C_num)

# In forward()
x_num = self.input_norm(x_num.transpose(1, 2)).transpose(1, 2)
```

### 2. **Hyperparameter Optimization**

#### High Priority Changes:
```python
# Recommended hyperparameters
d_model = 128        # Increase from 64
num_layers = 4       # Increase from 3
nhead = 8           # Increase from 4
dropout = 0.2       # Increase from 0.1
lr = 5e-4           # Decrease from 1e-3
pooling = "cls"     # Change from "mean"
```

#### Learning Rate Schedule:
```python
# In configure_optimizers()
optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
return [optimizer], [scheduler]
```

### 3. **Data Preprocessing Improvements**

#### Add Data Normalization:
```python
# In ETTh1Dataset._load_data()
# After loading sequences
sequences = (sequences - sequences.mean(axis=(0, 1))) / (sequences.std(axis=(0, 1)) + 1e-8)
```

#### Improve Class Balance:
```python
# Add class weighting in loss function
class_weights = torch.tensor([1.0, 0.9, 0.7])  # Inversely proportional to class frequency
self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

### 4. **Model Configuration Changes**

#### Patch-based Processing:
```python
# Add patch-based processing similar to PatchTST
self.patch_size = 16
self.stride = 8
self.patch_proj = nn.Conv1d(C_num, d_model, kernel_size=patch_size, stride=stride)
```

#### Positional Encoding:
```python
# Add positional encoding for better temporal understanding
self.pos_encoding = nn.Parameter(torch.randn(1, T, d_model))

# In forward()
z = z + self.pos_encoding[:, :z.size(1), :]
```

## Experimental Plan

### Phase 1: Quick Fixes (Expected 5-10% improvement)
1. **Remove dual-patch encoding** for datasets without missing values
2. **Add input normalization**
3. **Increase d_model to 128**
4. **Use class weighting**

### Phase 2: Architecture Improvements (Expected 10-15% improvement)
1. **Add positional encoding**
2. **Implement CLS token pooling**
3. **Add learning rate scheduling**
4. **Increase model depth (num_layers=4)**

### Phase 3: Advanced Optimizations (Expected 5-8% improvement)
1. **Implement patch-based processing**
2. **Add residual connections**
3. **Experiment with different sequence lengths**
4. **Add gradient clipping**

## Expected Performance Targets

| Phase | Target Accuracy | Key Changes |
|-------|----------------|-------------|
| Baseline | 74.4% | Current performance |
| Phase 1 | 80-85% | Architecture fixes + normalization |
| Phase 2 | 85-90% | Enhanced architecture + training |
| Phase 3 | 90-93% | Advanced optimizations |

## Implementation Priority

1. **CRITICAL**: Fix dual-patch encoding for datasets without missing values
2. **HIGH**: Add input normalization and class weighting
3. **HIGH**: Increase model capacity (d_model, num_layers, nhead)
4. **MEDIUM**: Add positional encoding and CLS token
5. **LOW**: Implement patch-based processing

## Code Changes Required

### Immediate fixes needed in `/home/ubuntu/DuET/duet/models/transformer.py`:
- Add conditional dual-patch encoding
- Add input normalization layer
- Increase default hyperparameters
- Add class weighting support

### Training script improvements in `/home/ubuntu/DuET/examples/compare_models_etth1.py`:
- Add data normalization
- Implement learning rate scheduling
- Add gradient clipping
- Better early stopping criteria

The CNN's superior performance (93.1% vs 74.4%) suggests that the current DuET architecture is not well-suited for this specific time series classification task. The suggested improvements should significantly close this performance gap.