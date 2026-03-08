# RTX 4090 训练指南

**GPU**: NVIDIA GeForce RTX 4090 (24GB GDDR6X)  
**状态**: ✅ 已优化配置

---

## 🚀 快速开始

### 1. 准备数据集

```bash
# 下载 LFW (评估用，187MB)
python3 tools/prepare_datasets.py --dataset lfw

# 下载 WIDER Face (检测训练，1GB)
python3 tools/prepare_datasets.py --dataset widerface

# 下载 CASIA-WebFace (识别训练，10GB)
# 或使用 WebFace12M (100GB)
```

### 2. 开始训练

```bash
# 赋予执行权限
chmod +x scripts/train_4090.sh

# 检测模型训练 (30-60 分钟)
./scripts/train_4090.sh detection datasets/widerface 0 32 100

# 识别模型训练 (1-2 天)
./scripts/train_4090.sh recognition datasets/webface12m 0,1,2,3 64 120
```

### 3. 评估与导出

```bash
# LFW 评估
python3 tools/evaluate.py \
  --checkpoint checkpoints/recognition/best.pth \
  --dataset lfw \
  --data-root datasets/lfw

# ONNX 导出
python3 tools/export_model.py onnx \
  --model detector \
  --checkpoint checkpoints/detection/best.pth

# TensorRT 优化
python3 tools/export_model.py trt \
  --onnx checkpoints/detection/model.onnx \
  --precision fp16
```

---

## 📊 RTX 4090 性能预估

### 训练速度

| 模型 | 数据集 | Batch Size | 预计时间 |
|-----|-------|-----------|---------|
| DKGA-Det | WIDER Face | 32 | 30-60 分钟 |
| DKGA-Det | 完整训练 | 32 | 2-4 小时 |
| DDFD-Rec | CASIA-WebFace | 32 | 1-2 小时 |
| DDFD-Rec | WebFace12M | 64 | 8-12 小时 |
| DDFD-Rec | 完整训练 | 64 | 1-2 天 |

### 推理速度

| 操作 | CPU (M1) | RTX 4090 | 提升 |
|-----|---------|---------|------|
| 检测推理 | 3170ms | ~3ms | **1000x** |
| 识别推理 | 75ms | ~2ms | **37x** |
| 检测训练 | 1.5 秒/iter | ~0.1 秒/iter | **15x** |
| 识别训练 | 0.5 秒/iter | ~0.03 秒/iter | **16x** |

---

## ⚙️ 多 GPU 配置

### 单卡训练
```bash
./scripts/train_4090.sh detection datasets/widerface 0 32 100
```

### 多卡训练 (推荐)
```bash
# 4 卡训练识别模型
./scripts/train_4090.sh recognition datasets/webface12m 0,1,2,3 64 120

# 8 卡训练
./scripts/train_4090.sh recognition datasets/webface12m 0,1,2,3,4,5,6,7 128 120
```

### 分布式训练配置

```bash
# 使用 torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train_recognition.py \
    --config configs/recognition/train.yaml \
    --data-dir datasets/webface12m \
    --ddp
```

---

## 📈 推荐训练配置

### 检测模型 (DKGA-Det)

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| Batch Size | 32 | 单卡 |
| Learning Rate | 0.001 | AdamW |
| Epochs | 100 | 完整训练 |
| Input Size | 640×640 | 标准输入 |
| 预计时间 | 2-4 小时 | 单卡 4090 |

### 识别模型 (DDFD-Rec)

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| Batch Size | 64 | 单卡 |
| Learning Rate | 0.1 | SGD |
| Epochs | 120 | 完整训练 |
| Input Size | 112×112 | 标准输入 |
| 预计时间 | 1-2 天 | 单卡 4090 |

---

## 🎯 训练监控

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 访问 http://localhost:6006
```

### 训练日志

```bash
# 实时查看训练日志
tail -f logs/detection/*.log
tail -f logs/recognition/*.log
```

### 检查点管理

```bash
# 查看检查点
ls -lh checkpoints/detection/
ls -lh checkpoints/recognition/

# 复制最佳权重
cp checkpoints/recognition/epoch_100.pth checkpoints/recognition/best.pth
```

---

## 📊 预期性能指标

### 检测模型 (WIDER Face)

| 子集 | 预期 mAP |
|-----|---------|
| Easy | >96% |
| Medium | >95% |
| Hard | >91% |

### 识别模型

| 数据集 | 预期准确率 |
|-------|-----------|
| LFW | >99.6% |
| CPLFW | >95.5% |
| IJB-C (FAR=1e-4) | >96.0% |

---

## 🔧 故障排除

### CUDA Out of Memory

```bash
# 减小 batch size
./scripts/train_4090.sh detection datasets/widerface 0 16 100

# 或使用梯度累积
# 修改 configs/detection/train.yaml
# grad_accum_steps: 2
```

### 训练速度慢

```bash
# 检查 GPU 使用率
nvidia-smi

# 启用 cudnn benchmark
# 修改 configs/default.yaml
# device.benchmark: true
```

### 多 GPU 训练问题

```bash
# 检查 GPU 可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -c "import torch; print(torch.cuda.device_count())"

# 使用 NCCL 后端
# configs/detection/train.yaml
# ddp.backend: nccl
```

---

## 📞 总结

**RTX 4090 优势**:
- ✅ 24GB GDDR6X 显存
- ✅ 16384 CUDA 核心
- ✅ 训练速度比 M1 快 15-1000x
- ✅ 支持完整数据集训练

**推荐流程**:
1. 下载完整数据集 (WIDER Face + WebFace12M)
2. 使用 `train_4090.sh` 开始训练
3. 监控训练进度 (TensorBoard)
4. 评估性能 (LFW/CPLFW)
5. 导出优化模型 (ONNX/TensorRT)

---

**最后更新**: 2026 年 3 月 7 日  
**状态**: ✅ RTX 4090 优化配置完成
