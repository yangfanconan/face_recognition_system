# DDFD-FaceRec 快速开始指南

**版本**: v1.0-alpha  
**更新日期**: 2026 年 3 月 7 日

---

## 🚀 5 分钟快速开始

### 1. 环境配置

```bash
# 进入项目目录
cd /Users/yangfan/face_recognition_system

# 激活虚拟环境
source venv/bin/activate

# 验证环境
python tools/check_env.py
```

### 2. 测试推理功能

```bash
# 运行推理测试
python tools/test_inference.py
```

预期输出:
```
============================================================
DDFD-FaceRec 推理测试
============================================================
测试人脸检测器
  检测人脸数：4
  平均推理时间：3170ms
测试人脸识别器
  特征维度：(409,)
  平均推理时间：75ms
测试特征匹配器
  相似度：0.1527
  自相似度：1.0000
============================================================
✅ 所有测试通过!
```

### 3. 下载数据集

```bash
# 下载 LFW (评估用)
python tools/prepare_datasets.py --dataset lfw --output-dir datasets

# 下载 WIDER Face (检测训练用)
python tools/prepare_datasets.py --dataset widerface --output-dir datasets
```

### 4. 开始训练

```bash
# 训练检测模型
./scripts/train.sh --model detection --dataset datasets/widerface

# 训练识别模型
./scripts/train.sh --model recognition --dataset datasets/webface12m
```

### 5. 启动 API 服务

```bash
# 启动服务
python -m api.main

# 访问 API 文档
# http://localhost:8000/docs
```

---

## 📋 详细使用指南

### 环境要求

- Python >= 3.10
- PyTorch >= 2.1
- CUDA >= 12.0 (可选，GPU 加速)
- 10GB+ 可用磁盘空间

### 安装步骤

#### Linux/Mac

```bash
# 克隆项目
git clone https://github.com/yangfanconan/face_recognition_system.git
cd face_recognition_system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

#### Windows

```bash
# 克隆项目
git clone https://github.com/yangfanconan/face_recognition_system.git
cd face_recognition_system

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### Docker 部署

```bash
# 构建镜像
docker build -t ddfd-face-rec:latest .

# 运行容器
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  ddfd-face-rec:latest
```

---

## 🔧 工具脚本

### 数据集准备

```bash
# 列出可用数据集
python tools/prepare_datasets.py --list

# 下载 LFW
python tools/prepare_datasets.py --dataset lfw

# 下载 WIDER Face
python tools/prepare_datasets.py --dataset widerface

# 下载所有
python tools/prepare_datasets.py --dataset all
```

### 模型训练

```bash
# 检测模型训练
./scripts/train.sh --model detection \
  --dataset datasets/widerface \
  --gpus 0 \
  --batch-size 32 \
  --epochs 100

# 识别模型训练 (多 GPU)
./scripts/train.sh --model recognition \
  --dataset datasets/webface12m \
  --gpus 0,1,2,3 \
  --batch-size 64 \
  --epochs 120

# 恢复训练
./scripts/train.sh --model detection \
  --resume checkpoints/detection/epoch_50.pth
```

### 模型导出

```bash
# 导出检测模型为 ONNX
python tools/export_model.py onnx \
  --model detector \
  --checkpoint checkpoints/detection/best.pth \
  --output checkpoints/detection/model.onnx

# 导出识别模型为 ONNX
python tools/export_model.py onnx \
  --model recognizer \
  --checkpoint checkpoints/recognition/best.pth \
  --output checkpoints/recognition/model.onnx

# 构建 TensorRT 引擎 (FP16)
python tools/export_model.py trt \
  --onnx checkpoints/detection/model.onnx \
  --output checkpoints/detection/model.trt \
  --precision fp16
```

### 模型评估

```bash
# LFW 评估
python tools/evaluate.py \
  --checkpoint checkpoints/recognition/best.pth \
  --dataset lfw \
  --data-root datasets/lfw \
  --output results/lfw_eval.json

# CPLFW 评估
python tools/evaluate.py \
  --checkpoint checkpoints/recognition/best.pth \
  --dataset cplfw \
  --data-root datasets/cplfw
```

---

## 📊 API 使用示例

### 人脸检测

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "image=@test.jpg"
```

响应:
```json
{
  "success": true,
  "data": {
    "faces": [
      {
        "bbox": [100, 150, 200, 250],
        "score": 0.998,
        "landmarks": [[x1,y1], [x2,y2], ...]
      }
    ],
    "count": 1
  },
  "inference_time_ms": 4.2
}
```

### 特征提取

```bash
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "image=@face.jpg"
```

### 人脸验证

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": [...],
    "feature2": [...],
    "threshold": 0.6
  }'
```

### 人脸搜索

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "feature": [...],
    "top_k": 10,
    "threshold": 0.6
  }'
```

---

## 🐛 常见问题

### Q: CUDA 不可用怎么办？

A: 模型将在 CPU 上运行，速度会较慢。确保已安装正确版本的 PyTorch:
```bash
pip install torch torchvision torchaudio  # CPU 版本
# 或
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120  # GPU 版本
```

### Q: 内存不足怎么办？

A: 减小 batch size:
```bash
./scripts/train.sh --model detection --batch-size 8
```

### Q: 如何恢复训练？

A: 使用 `--resume` 参数:
```bash
./scripts/train.sh --model detection --resume checkpoints/detection/epoch_50.pth
```

### Q: 如何查看训练进度？

A: 使用 TensorBoard:
```bash
tensorboard --logdir logs/
# 访问 http://localhost:6006
```

---

## 📞 获取帮助

- **GitHub Issues**: https://github.com/yangfanconan/face_recognition_system/issues
- **技术文档**: `PROJECT_DESIGN.md`, `ACADEMIC_SUMMARY.md`
- **快速参考**: 本文件

---

**最后更新**: 2026 年 3 月 7 日  
**版本**: v1.0-alpha
