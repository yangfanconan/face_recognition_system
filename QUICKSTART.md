# DDFD-FaceRec 快速开始指南

本指南帮助您快速完成环境配置并开始使用 DDFD-FaceRec。

---

## 📋 前置要求

- Python >= 3.10
- CUDA >= 12.0 (可选，用于 GPU 加速)
- 10GB+ 可用磁盘空间

---

## 🚀 快速安装 (5 分钟)

### 1. 克隆/进入项目目录

```bash
cd /Users/yangfan/face_recognition_system
```

### 2. 运行环境配置脚本

**Linux/Mac:**
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

**Windows:**
```bash
scripts\setup_env.bat
```

### 3. 激活虚拟环境

```bash
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

---

## ✅ 验证安装

### 检查环境配置

```bash
python tools/check_env.py
```

预期输出:
```
============================================================
DDFD-FaceRec 训练环境检查
============================================================
Python 版本：3.10.x  ✅
PyTorch 版本：2.1.x  ✅
CUDA 可用：True     ✅
...
✅ 所有检查通过！
```

### 运行推理测试

```bash
python tools/test_inference.py
```

预期输出:
```
============================================================
DDFD-FaceRec 推理测试
============================================================
测试人脸检测器...
  平均推理时间：4.2ms
测试人脸识别器...
  平均推理时间：3.8ms
...
✅ 所有测试通过!
```

---

## 📥 下载数据集 (可选)

### 下载 LFW (评估用，免费)

```bash
python tools/download_datasets.py --dataset lfw
```

### 下载 WIDER FACE (检测用，免费)

```bash
python tools/download_datasets.py --dataset widerface
```

### 下载大型数据集 (需要申请)

```bash
# VGGFace2 - 需要申请
python tools/download_datasets.py --dataset vggface2 --auth-url <URL>

# WebFace12M - 需要申请
python tools/download_datasets.py --dataset webface12m --auth-url <URL>
```

---

## 🏋️ 训练模型

### 训练检测模型

```bash
python tools/train_detection.py \
  --config configs/detection/train.yaml \
  --data-dir datasets/widerface \
  --output-dir checkpoints/detection
```

### 训练识别模型

```bash
python tools/train_recognition.py \
  --config configs/recognition/train.yaml \
  --data-dir datasets/webface12m \
  --output-dir checkpoints/recognition
```

### 微调 (针对特定场景)

```bash
# 口罩人脸微调
python tools/finetune.py \
  --scenario mask \
  --checkpoint checkpoints/recognition/best.pth \
  --data-dir datasets/masked_faces
```

---

## 📊 评估模型

### LFW 评估

```bash
python tools/evaluate.py \
  --checkpoint checkpoints/recognition/best.pth \
  --dataset lfw \
  --data-root datasets/lfw \
  --output results/lfw_eval.json
```

### CPLFW 评估

```bash
python tools/evaluate.py \
  --checkpoint checkpoints/recognition/best.pth \
  --dataset cplfw \
  --data-root datasets/cplfw
```

---

## 🌐 启动 API 服务

### 启动服务

```bash
python -m api.main
```

服务将在 `http://localhost:8000` 启动

### 访问 API 文档

打开浏览器访问：`http://localhost:8000/docs`

### API 使用示例

**人脸检测:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "image=@test.jpg"
```

**特征提取:**
```bash
curl -X POST "http://localhost:8000/api/v1/extract" \
  -F "image=@face.jpg"
```

**人脸比对:**
```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "Content-Type: application/json" \
  -d '{"feature1": [...], "feature2": [...]}'
```

---

## 🐳 Docker 部署

### 构建并运行

```bash
docker-compose up -d
```

### 访问服务

| 服务 | 地址 |
|-----|------|
| API | http://localhost:8000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| MinIO | http://localhost:9000 |

### 查看日志

```bash
docker-compose logs -f api
```

### 停止服务

```bash
docker-compose down
```

---

## 📁 目录结构

```
face_recognition_system/
├── configs/           # 配置文件
├── models/            # 模型定义
├── inference/         # 推理模块
├── tools/             # 工具脚本
├── tests/             # 测试
├── checkpoints/       # 模型权重
├── logs/              # 日志
└── datasets/          # 数据集
```

---

## 🔧 常见问题

### Q: CUDA 不可用怎么办？

A: 如果没有 GPU，模型将在 CPU 上运行，速度会较慢。确保已安装正确版本的 PyTorch:
```bash
pip install torch torchvision torchaudio  # CPU 版本
```

### Q: 内存不足怎么办？

A: 减小 batch size:
```bash
# 修改配置文件或使用命令行覆盖
python tools/train_detection.py --config configs/detection/train.yaml \
  --override training.batch_size=8
```

### Q: 如何恢复训练？

A: 使用 `--resume` 参数:
```bash
python tools/train_detection.py \
  --config configs/detection/train.yaml \
  --resume checkpoints/detection/epoch_50.pth
```

### Q: 如何查看训练进度？

A: 使用 TensorBoard:
```bash
tensorboard --logdir logs/
# 访问 http://localhost:6006
```

---

## 📞 获取帮助

- 查看完整文档：`PROJECT_DESIGN.md`
- 查看技术细节：`FINAL_SUMMARY.md`
- 报告问题：创建 GitHub Issue

---

**最后更新**: 2026 年 3 月 7 日
