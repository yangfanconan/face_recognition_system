#!/bin/bash

# ============================================
# Apple M1 优化训练脚本
# ============================================

set -e

echo "========================================"
echo "DDFD-FaceRec M1 优化训练"
echo "========================================"
echo ""
echo "硬件配置:"
echo "  CPU: Apple M1 (8 核)"
echo "  GPU: Apple M1 (8 核)"
echo "  内存：8GB"
echo ""

# 启用 MPS 加速
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 检查 MPS 支持
echo "检查 PyTorch MPS 支持..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✅ 使用 MPS 加速')
    device = 'mps'
else:
    print('⚠️  MPS 不可用，使用 CPU')
    device = 'cpu'
"

echo ""
echo "========================================"
echo "M1 优化配置"
echo "========================================"
echo ""
echo "推荐配置:"
echo "  检测模型: batch_size=4, epochs=50"
echo "  识别模型: batch_size=4, epochs=80"
echo "  预计时间: 检测 4-6h, 识别 8-12h"
echo ""
echo "训练命令:"
echo ""
echo "# 检测模型训练"
echo "./scripts/train_m1.sh --model detection --dataset datasets/widerface"
echo ""
echo "# 识别模型训练"
echo "./scripts/train_m1.sh --model recognition --dataset datasets/casia_webface"
echo ""

# 解析参数
MODEL="${1:-detection}"
DATASET="${2:-datasets/widerface}"

case $MODEL in
    detection)
        echo "开始训练检测模型..."
        python3 tools/train_detection.py \
            --config configs/detection/train.yaml \
            --data-dir "$DATASET" \
            --output-dir checkpoints/detection \
            --batch-size 4 \
            --epochs 50
        ;;
    recognition)
        echo "开始训练识别模型..."
        python3 tools/train_recognition.py \
            --config configs/recognition/train.yaml \
            --data-dir "$DATASET" \
            --output-dir checkpoints/recognition \
            --batch-size 4 \
            --epochs 80
        ;;
    *)
        echo "未知模型：$MODEL"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "训练完成!"
echo "========================================"
