# Apple M1 使用指南

**设备**: Apple M1 (8GB 内存)  
**状态**: ✅ 已优化配置

---

## 📊 您的配置分析

| 组件 | 配置 | 训练适用性 |
|-----|------|-----------|
| CPU | Apple M1 (8 核) | ✅ 优秀 |
| GPU | Apple M1 (8 核) | ✅ 良好 (MPS) |
| 内存 | 8GB | ⚠️ 偏少 (需小 batch) |
| 磁盘 | 228GB (19GB 可用) | ⚠️ 需清理 |

---

## 🚀 M1 优化方案

### 1. 启用 MPS 加速

```bash
# 检查 MPS 支持
python3 -c "import torch; print(torch.backends.mps.is_available())"

# 如果返回 True，可以使用 MPS 加速
# 如果返回 False，将自动回退到 CPU
```

### 2. 使用优化脚本训练

```bash
# 赋予执行权限
chmod +x scripts/train_m1.sh

# 训练检测模型 (4-6 小时)
./scripts/train_m1.sh detection datasets/widerface

# 训练识别模型 (8-12 小时)
./scripts/train_m1.sh recognition datasets/casia_webface
```

### 3. 内存优化配置

| 模型 | Batch Size | 内存占用 | 训练时间 |
|-----|-----------|---------|---------|
| 检测 | 4 | ~6GB | 4-6 小时 |
| 检测 | 2 | ~4GB | 6-8 小时 |
| 识别 | 4 | ~7GB | 8-12 小时 |
| 识别 | 2 | ~5GB | 12-16 小时 |

---

## 💾 磁盘空间管理

### 当前状态
```
总空间：228GB
已使用：15GB
可用：19GB
```

### 数据集大小
| 数据集 | 大小 | 建议 |
|-------|------|------|
| LFW | 187MB | ✅ 可下载 |
| WIDER Face | 1GB | ✅ 可下载 |
| CASIA-WebFace | 10GB | ⚠️ 需清理 |
| WebFace12M | 100GB | ❌ 空间不足 |

### 清理建议

```bash
# 1. 清理系统缓存
sudo rm -rf ~/.cache/pip
sudo rm -rf ~/Library/Caches

# 2. 清理 Xcode 缓存
sudo rm -rf ~/Library/Developer/Xcode/DerivedData

# 3. 清理 Docker (如果有)
docker system prune -a

# 4. 清理下载文件夹
rm -rf ~/Downloads/*.dmg
rm -rf ~/Downloads/*.zip
```

---

## 📈 预期性能

### 训练速度 (M1 8GB)

| 操作 | CPU | MPS (GPU) |
|-----|-----|----------|
| 检测推理 | 3170ms | ~1500ms |
| 识别推理 | 75ms | ~40ms |
| 检测训练 | 1.5 秒/iter | ~0.8 秒/iter |
| 识别训练 | 0.5 秒/iter | ~0.3 秒/iter |

### 与高端 GPU 对比

| 设备 | 检测训练 | 识别训练 |
|-----|---------|---------|
| **Apple M1** | 4-6 小时 | 8-12 小时 |
| RTX 3090 | 30 分钟 | 2-3 小时 |
| A100 (8×) | 10 分钟 | 30 分钟 |

**结论**: M1 速度约为 RTX 3090 的 1/8，但完全可以用于开发和测试！

---

## 🎯 推荐方案

### 方案 A: 本地训练 (推荐用于开发)

**适合**: 模型开发、调试、小规模测试

```bash
# 1. 下载小数据集
python3 tools/prepare_datasets.py --dataset lfw
python3 tools/prepare_datasets.py --dataset widerface

# 2. 快速测试训练 (1 epoch)
python3 tools/train_detection.py \
    --config configs/detection/train.yaml \
    --data-dir datasets/widerface \
    --epochs 1 \
    --batch-size 2

# 3. 完整训练
./scripts/train_m1.sh detection datasets/widerface
```

### 方案 B: 云端训练 (推荐用于生产)

**适合**: 最终模型训练、性能验证

**推荐**:
- **Google Colab Pro**: $10/月 (A100 GPU)
- **Kaggle**: 免费 (P100 GPU, 每周 30 小时)

**步骤**:
1. 上传项目到 Colab
2. 挂载 Google Drive
3. 运行训练脚本
4. 下载训练好的权重

### 方案 C: 使用预训练模型

**适合**: 快速部署、无需训练

我可以帮您:
1. 查找开源预训练权重
2. 转换格式适配项目
3. 直接部署使用

---

## ⚡ 立即开始

### 快速测试 (5 分钟)

```bash
# 运行自动测试
python3 scripts/auto_train.py
```

### 本地训练 (4-6 小时)

```bash
# 1. 准备数据
python3 tools/prepare_datasets.py --dataset widerface

# 2. 开始训练
./scripts/train_m1.sh detection datasets/widerface

# 3. 等待完成...
# 可以在另一个终端监控进度
tail -f logs/detection/*.log
```

---

## 📞 总结

**您的 M1 配置**:
- ✅ **可以训练**，不是性能不行
- ⚠️ 需要优化配置 (小 batch size)
- ⚠️ 训练时间较长 (但可接受)
- ⚠️ 内存 8GB 是主要限制

**建议**:
1. 先用小数据集测试 (LFW, WIDER Face)
2. 使用 `scripts/train_m1.sh` 优化脚本
3. 如果需要快速训练，考虑云端 GPU
4. 立即可用预训练模型部署

---

**最后更新**: 2026 年 3 月 7 日  
**状态**: ✅ M1 优化配置完成
