# model_train.py vs model_train_ver1207.py 核心差异对比

## 一、导入和依赖差异

### model_train.py
- 导入 `pathlib`（用于路径处理）
- 导入 `torch.nn`（定义 WeightedLoss 类）
- **在文件内定义 WeightedLoss 类**

### model_train_ver1207.py
- 导入 `Path` from `pathlib`
- 导入 `TimeSeries` from `darts`
- **从 model_train 导入 WeightedLoss 类**（复用）
- 添加 CUDA Tensor Core 优化设置

---

## 二、参数解析 (parse_args) 差异

| 参数 | model_train.py | model_train_ver1207.py | 影响 |
|------|----------------|------------------------|------|
| `--data` | 默认值 `"Dataset/ts_data.pkl"` | **必需参数** (`required=True`) | ⚠️ 必须显式指定 |
| `--lambda` | 使用 `--lambda` | 使用 `--lambda_weight` | 参数名不同 |
| `--lr` | 默认值 `3e-4` | **必需参数** (`required=True`) | ⚠️ 必须显式指定 |
| `--hidden_size` | 默认值 `32` | **必需参数** (`required=True`) | ⚠️ 必须显式指定 |
| `--ff_size` | 默认值 `64` | **必需参数** (`required=True`) | ⚠️ 必须显式指定 |
| `--dropout` | 默认值 `0.1` | **必需参数** (`required=True`) | ⚠️ 必须显式指定 |
| `--epochs` | 默认值 `200` | 默认值 `50` | ⚠️ 默认训练轮数不同 |
| `--lr_scheduler` | ✅ 支持（默认 `exponential`） | ❌ **不支持** | 🔴 **重要差异** |
| `--lr_gamma` | ✅ 支持（默认 `0.99`） | ❌ **不支持** | 🔴 **重要差异** |
| `--grad_clip` | ✅ 支持（默认 `0.5`） | ❌ **不支持** | 🔴 **重要差异** |
| `--covariate_mode` | ✅ 支持（`none`/`lagged`） | ❌ **不支持** | 🔴 **重要差异** |
| `--combine_train_val` | ❌ 不支持 | ✅ **新增功能** | 🟢 新功能 |
| `--model_path` | 参数名 | `--output_model` | 参数名不同 |

---

## 三、数据处理差异

### model_train.py
```python
# 使用 _prepare_covariates 处理协变量
train_covs = _cast_series_list(_prepare_covariates(dataset["train"]["cov"]))
val_covs = _cast_series_list(_prepare_covariates(dataset["val"]["cov"]))

# 检查是否有验证集
has_val = any(ts is not None for ts in val_targets)
has_val_cov = any(cov is not None for cov in val_covs)
```

### model_train_ver1207.py
```python
# 直接使用协变量，不经过 _prepare_covariates
train_covs = _cast_series_list(dataset["train"]["cov"])
val_covs = _cast_series_list(dataset["val"]["cov"])

# 支持合并训练集和验证集
if args.combine_train_val:
    # 合并逻辑...
```

**差异影响：**
- `model_train.py` 对协变量有额外处理（可能处理 None 值）
- `model_train_ver1207.py` 支持合并 train+val 进行最终训练

---

## 四、设备检测差异

### model_train.py
```python
if torch.cuda.is_available():
    accelerator = "gpu"
    devices = "auto"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    accelerator = "mps"  # ✅ 支持 Apple Silicon
    devices = "auto"
else:
    accelerator = "cpu"
    devices = 1
```

### model_train_ver1207.py
```python
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = "auto" if accelerator == "gpu" else 1
# ❌ 不支持 Apple Silicon MPS
```

**差异影响：** 🔴 在 Mac 上可能无法使用 MPS 加速

---

## 五、学习率调度器差异

### model_train.py
```python
if args.lr_scheduler == "exponential":
    lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    lr_scheduler_kwargs = {"gamma": args.lr_gamma}
else:
    lr_scheduler_cls = None
    lr_scheduler_kwargs = None

# 在模型中使用
lr_scheduler_cls=lr_scheduler_cls,
lr_scheduler_kwargs=lr_scheduler_kwargs,
```

### model_train_ver1207.py
```python
# ❌ 完全不支持学习率调度器
# 固定学习率训练
```

**差异影响：** 🔴 **重要差异** - 学习率不会衰减，可能影响收敛

---

## 六、Early Stopping 配置差异

| 配置项 | model_train.py | model_train_ver1207.py |
|--------|----------------|------------------------|
| `patience` | `10` | `5` |
| `min_delta` | `1e-5` | 未设置（使用默认值） |

**差异影响：** ⚠️ 早停策略更宽松（需要更多轮次才停止）

---

## 七、Callbacks 差异

### model_train.py
```python
try:
    from darts.utils.callbacks import TFMProgressBar
    progress_bar = TFMProgressBar(
        enable_sanity_check_bar=False, enable_validation_bar=False
    )
    callbacks = [early_stopper, progress_bar]  # ✅ 包含进度条
except Exception:
    callbacks = [early_stopper]
```

### model_train_ver1207.py
```python
callbacks = [early_stop]  # ❌ 只有早停，无进度条
```

**差异影响：** 进度显示方式不同

---

## 八、模型配置差异

| 配置项 | model_train.py | model_train_ver1207.py |
|--------|----------------|------------------------|
| `log_tensorboard` | ✅ `True` | ❌ 未设置 |
| `add_encoders` | ✅ `{"cyclic": {"future": ["dayofweek"]}}` | ❌ 未设置 |
| `save_checkpoints` | 未设置（可能使用默认） | ✅ `False` |
| `force_reset` | 未设置 | ✅ `True` |
| `gradient_clip_val` | ✅ 在 `pl_trainer_kwargs` 中设置 | ❌ 未设置 |
| `max_epochs` | ✅ 在 `pl_trainer_kwargs` 中设置 | ❌ 未设置 |

**差异影响：**
- 🔴 **无梯度裁剪** - 可能导致梯度爆炸
- 🔴 **无 TensorBoard 日志** - 无法可视化训练过程
- 🔴 **无时间编码器** - 可能丢失时间特征

---

## 九、训练过程差异

### model_train.py
```python
model.fit(
    series=train_targets,
    past_covariates=train_covs,
    val_series=val_targets if has_val else None,
    val_past_covariates=val_covs if has_val_cov else None,
    epochs=args.epochs,
    dataloader_kwargs={"batch_size": args.batch_size},
    verbose=False,  # ❌ 不显示详细信息
)
```

### model_train_ver1207.py
```python
model.fit(
    series=fit_targets,  # 可能是合并后的数据
    past_covariates=fit_covs,
    val_series=fit_val_targets,  # 如果合并则为 None
    val_past_covariates=fit_val_covs,
    epochs=args.epochs,
    dataloader_kwargs={"batch_size": args.batch_size},
    verbose=True,  # ✅ 显示详细信息
)
```

**差异影响：**
- 如果使用 `--combine_train_val`，验证集会被合并到训练集，无法进行验证

---

## 十、PyTorch 默认数据类型设置

### model_train.py
```python
torch.set_default_dtype(torch.float32)  # ✅ 显式设置
```

### model_train_ver1207.py
```python
# ❌ 未设置，使用系统默认
```

---

## 十一、其他差异

### model_train.py
- 使用 `pathlib.Path` 处理路径
- 更详细的文档字符串

### model_train_ver1207.py
- 添加 CUDA Tensor Core 优化
- 打印最佳超参数信息

---

## 总结：输出结果差异评估

### 🔴 **会有显著差异的方面：**

1. **学习率调度** - `model_train.py` 使用指数衰减，`model_train_ver1207.py` 固定学习率
   - **影响：** 收敛速度和最终性能可能不同

2. **梯度裁剪** - `model_train.py` 有，`model_train_ver1207.py` 无
   - **影响：** 训练稳定性可能不同，可能出现梯度爆炸

3. **时间编码器** - `model_train.py` 有周期性编码，`model_train_ver1207.py` 无
   - **影响：** 模型可能无法学习时间周期性特征

4. **TensorBoard 日志** - `model_train.py` 有，`model_train_ver1207.py` 无
   - **影响：** 无法可视化训练过程（不影响模型性能）

5. **早停策略** - `patience` 和 `min_delta` 不同
   - **影响：** 训练轮数可能不同

6. **协变量处理** - 处理方式不同
   - **影响：** 如果使用协变量，行为可能不同

### ⚠️ **可能有差异的方面：**

1. **默认训练轮数** - 50 vs 200
2. **设备支持** - MPS 支持差异
3. **数据合并选项** - `model_train_ver1207.py` 支持合并 train+val

### ✅ **相同或类似的方面：**

1. 核心模型架构（TSMixer）
2. 损失函数（WeightedLoss）
3. 基本训练流程
4. 数据加载方式

---

## 建议

如果要让 `model_train_ver1207.py` 产生与 `model_train.py` 相似的结果，需要：

1. ✅ 添加学习率调度器支持
2. ✅ 添加梯度裁剪
3. ✅ 添加时间编码器
4. ✅ 添加 MPS 支持
5. ✅ 调整早停参数
6. ✅ 添加 TensorBoard 日志
7. ✅ 设置 PyTorch 默认数据类型

