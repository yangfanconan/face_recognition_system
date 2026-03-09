# 人脸识别系统优化方案 - 实施总结

**文档版本**: v1.0  
**创建日期**: 2026 年 3 月 9 日  
**基于测试**: LFW AUC=0.5773, EER=0.4385, 准确率=66.62%

---

## 📋 优化目标

| 指标 | 当前值 | 目标值 | 提升幅度 |
|-----|--------|--------|---------|
| **识别 AUC** | 0.5773 | ≥0.95 | +65% |
| **识别 EER** | 0.4385 | ≤0.05 | -89% |
| **识别准确率** | 66.62% | ≥95% | +43% |
| **最佳阈值** | 0.9883 | 0.4-0.6 | 回归正常 |
| **检测 mAP** | N/A | ≥80% | - |

---

## ✅ 已完成优化内容

### 1. 识别模型优化

#### 1.1 训练策略优化

**文件**: `tools/train_recognition_v2.py`

**优化点**:
- ✅ 增量训练支持（从检查点恢复）
- ✅ 余弦退火学习率调度
- ✅ 早停策略（patience=10）
- ✅ 混合精度训练（AMP）
- ✅ TensorBoard 日志

**关键超参数**:
```yaml
epochs: 50                    # 从 5 增加到 50
batch_size: 64                # 从 16 增加到 64
lr: 0.1                       # 初始学习率
lr_scheduler: CosineAnnealingLR # 余弦退火
T_max: 50
eta_min: 1e-6
```

#### 1.2 损失函数优化

**文件**: `models/recognition/losses.py`

**新增损失函数**:
- ✅ `ArcFaceLoss` - 角度边界损失（推荐）
- ✅ `CosFaceLoss` - 余弦边界损失
- ✅ `AMSoftmaxLoss` - 加性边界 Softmax
- ✅ `FocalLoss` - 类别不均衡处理

**推荐配置**:
```python
criterion = ArcFaceLoss(
    in_features=512,
    out_features=num_classes,
    margin=0.5,      # 角度边界
    scale=30         # 特征缩放
)
```

#### 1.3 数据增强

**文件**: `data/transforms/face_augmentation.py`

**增强策略**:
- ✅ 水平翻转 (p=0.5)
- ✅ 随机旋转 (±15°)
- ✅ 光照变化 (亮度/对比度/饱和度)
- ✅ 颜色扰动 (色相/饱和度/值)
- ✅ 噪声增强 (高斯噪声)
- ✅ 随机遮挡 (CoarseDropout)

---

### 2. 检测模型修复

#### 2.1 后处理修复

**文件**: `models/detection/post_process.py`

**修复内容**:
- ✅ `decode_bbox_fixed()` - 修复坐标解码
  - 限制偏移量范围 (clamp -10 到 10)
  - 使用 exp() 确保宽高为正
  - 裁剪到图像范围
  
- ✅ `nms_fixed()` - 修复 NMS 实现
  - 正确的 IoU 计算
  - 置信度阈值过滤
  - 按类别批量 NMS

- ✅ 置信度校准
  - 添加 Sigmoid 激活
  - 确保输出在 [0, 1] 范围

---

### 3. 数据扩充方案

#### 3.1 CASIA-WebFace 融合

**文件**: `tools/data/prepare_casia_webface.py`

**数据集规模**:
| 数据集 | 身份数 | 图像数 |
|-------|--------|--------|
| LFW | 5,749 | 13,000+ |
| CASIA-WebFace | 10,575 | 494,414 |
| **合并后** | **16,324** | **507,414+** |

**使用步骤**:
```bash
# 1. 下载 CASIA-WebFace（需手动）
# 访问：https://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html

# 2. 解压并转换格式
python tools/data/prepare_casia_webface.py --mode all

# 3. 使用合并数据集训练
python tools/train_recognition_v2.py --data-dir datasets/lfw_merged
```

---

## 🚀 一键运行命令

### 识别模型训练

```bash
# 基础训练（50 epochs，ArcFace Loss）
python tools/train_recognition_v2.py \
  --data-dir datasets/lfw \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.1 \
  --loss arcface \
  --margin 0.5 \
  --scale 30 \
  --save-dir checkpoints/recognition_v2

# 增量训练（从现有模型继续）
python tools/train_recognition_v2.py \
  --data-dir datasets/lfw \
  --epochs 50 \
  --batch-size 64 \
  --loss arcface \
  --resume checkpoints/recognition/best.pth \
  --save-dir checkpoints/recognition_v2
```

### 检测模型修复验证

```bash
# 测试 NMS 修复
python models/detection/post_process.py

# 测试 bbox 解码修复
python -c "
from models.detection.post_process import test_decode_bbox
test_decode_bbox()
"
```

### 端到端验证

```bash
# 识别模型单独测试
python tests/benchmarks/lfw_recognition_only_test.py \
  --checkpoint checkpoints/recognition_v2/best.pth

# 端到端测试（检测 + 识别）
python tests/benchmarks/lfw_end_to_end_test.py \
  --detector_ckpt checkpoints/detection_v2/best.pth \
  --recognizer_ckpt checkpoints/recognition_v2/best.pth
```

---

## 📊 预期效果

### 训练进度预估

| Epochs | 预计 AUC | 预计 EER | 预计准确率 |
|--------|---------|---------|-----------|
| 5 (当前) | 0.58 | 0.44 | 67% |
| 20 | 0.75 | 0.25 | 80% |
| 50 | 0.90 | 0.10 | 90% |
| 100 | 0.95+ | <0.05 | 95%+ |

### 训练时间预估（RTX 4090）

| 数据集 | Batch Size | 50 Epochs | 100 Epochs |
|-------|-----------|----------|-----------|
| LFW (13k) | 64 | ~8 小时 | ~16 小时 |
| LFW+WebFace (500k) | 64 | ~3 天 | ~6 天 |

---

## 🔧 故障排查

### 问题 1: CUDA Out of Memory

**解决方案**:
```bash
# 减小 batch_size
python tools/train_recognition_v2.py \
  --batch-size 32 \  # 从 64 改为 32
  ...
```

### 问题 2: 损失不下降

**可能原因**:
- 学习率太大/太小
- 数据标注错误
- 模型结构问题

**解决方案**:
```bash
# 调整学习率
python tools/train_recognition_v2.py \
  --lr 0.01 \  # 从 0.1 改为 0.01
  ...
```

### 问题 3: 过拟合

**症状**: 训练准确率高，验证准确率低

**解决方案**:
```bash
# 增加数据增强
# 在 train_recognition_v2.py 中修改 train_transform

# 启用早停
# 已默认启用 (patience=10)

# 增加 Dropout
# 在模型中添加 Dropout 层
```

---

## 📁 文件清单

### 新增文件

| 文件 | 功能 | 行数 |
|-----|------|------|
| `OPTIMIZATION_PLAN.md` | 优化方案文档 | ~1500 |
| `tools/train_recognition_v2.py` | 增量训练脚本 | ~400 |
| `models/recognition/losses.py` | ArcFace/CosFace损失 | ~300 |
| `models/detection/post_process.py` | 检测后处理修复 | ~350 |
| `tools/data/prepare_casia_webface.py` | 数据准备脚本 | ~200 |
| `data/transforms/face_augmentation.py` | 数据增强 | ~200 |

### 修改文件

| 文件 | 修改内容 |
|-----|---------|
| `README.md` | 添加测试框架说明 |
| `tests/benchmarks/*` | 测试框架代码 |

---

## 📈 监控与日志

### TensorBoard 使用

```bash
# 启动 TensorBoard
tensorboard --logdir runs/recognition_v2

# 访问 http://localhost:6006 查看:
# - 训练/验证损失曲线
# - 准确率曲线
# - 学习率变化
# - 特征分布直方图
```

### 关键指标监控

训练过程中应关注:
- **训练损失**: 应持续下降
- **验证准确率**: 应持续上升
- **学习率**: 按余弦退火调度
- **早停计数器**: 连续未改善 epoch 数

---

## 🎯 下一步行动

### 立即执行（高优先级）

1. 🔴 **开始识别模型增量训练**
   ```bash
   python tools/train_recognition_v2.py \
     --data-dir datasets/lfw \
     --epochs 50 \
     --resume checkpoints/recognition/best.pth
   ```

2. 🔴 **验证检测后处理修复**
   ```bash
   python models/detection/post_process.py
   ```

### 短期执行（中优先级）

3. 🟡 **下载并融合 CASIA-WebFace**
   - 申请数据集访问权限
   - 运行 `prepare_casia_webface.py`

4. 🟡 **继续训练检测模型**
   - 应用后处理修复
   - 训练 50+ epochs

### 长期执行（低优先级）

5. 🟢 **模型优化与部署**
   - ONNX 转换
   - TensorRT 量化
   - 性能基准测试

---

## 📞 技术支持

- **优化方案文档**: `OPTIMIZATION_PLAN.md`
- **训练脚本**: `tools/train_recognition_v2.py`
- **损失函数**: `models/recognition/losses.py`
- **后处理修复**: `models/detection/post_process.py`

---

*最后更新*: 2026 年 3 月 9 日  
*预期完成时间*: 训练 50 epochs 后（约 8 小时）
