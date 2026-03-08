# DDFD-FaceRec 训练报告

**生成时间**: 2026-03-08 14:14:33

## 环境信息

- Python: 3.13.5
- PyTorch: 已安装
- CUDA: 不可用

## 数据集

- LFW 测试集：已创建 (模拟数据)
- WIDER Face 测试集：已创建 (模拟数据)

## 训练状态

- 检测模型：⏳ 待训练
- 识别模型：⏳ 待训练

## 下一步

1. 下载真实训练数据集
2. 开始完整训练
3. 性能评估
4. 模型导出

## 使用真实数据

```bash
# 下载 LFW
python3 tools/prepare_datasets.py --dataset lfw

# 下载 WIDER Face
python3 tools/prepare_datasets.py --dataset widerface

# 训练检测模型
./scripts/train.sh --model detection --dataset datasets/widerface

# 训练识别模型
./scripts/train.sh --model recognition --dataset datasets/webface12m
```
