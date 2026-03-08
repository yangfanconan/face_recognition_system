#!/bin/bash

# ============================================
# RTX 4090 优化训练脚本
# ============================================

set -e

echo "========================================"
echo "DDFD-FaceRec RTX 4090 训练"
echo "========================================"
echo ""

# 检查 GPU
echo "检查 GPU 配置..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print('✅ 使用 CUDA 加速')
else:
    print('❌ CUDA 不可用')
    exit(1)
"

echo ""
echo "========================================"
echo "RTX 4090 优化配置"
echo "========================================"
echo ""

# 解析参数
MODEL="${1:-detection}"
DATASET="${2:-datasets/widerface}"
GPUS="${3:-0}"
BATCH_SIZE="${4:-32}"
EPOCHS="${5:-100}"

echo "配置:"
echo "  模型：$MODEL"
echo "  数据集：$DATASET"
echo "  GPU: $GPUS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES="$GPUS"

case $MODEL in
    detection)
        echo "========================================"
        echo "训练检测模型 (DKGA-Det)"
        echo "========================================"
        echo ""
        echo "预计时间:"
        echo "  WIDER Face: 30-60 分钟"
        echo "  完整训练：2-4 小时"
        echo ""
        
        python3 tools/train_detection.py \
            --config configs/detection/train.yaml \
            --data-dir "$DATASET" \
            --output-dir checkpoints/detection \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --ddp
        ;;
    
    recognition)
        echo "========================================"
        echo "训练识别模型 (DDFD-Rec)"
        echo "========================================"
        echo ""
        echo "预计时间:"
        echo "  CASIA-WebFace: 1-2 小时"
        echo "  WebFace12M: 8-12 小时"
        echo "  完整训练：1-2 天"
        echo ""
        
        python3 tools/train_recognition.py \
            --config configs/recognition/train.yaml \
            --data-dir "$DATASET" \
            --output-dir checkpoints/recognition \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --ddp
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
echo ""
echo "检查点位置：checkpoints/$MODEL/"
echo ""
echo "下一步:"
echo "  1. 评估模型性能"
echo "     python3 tools/evaluate.py --checkpoint checkpoints/$MODEL/best.pth --dataset lfw"
echo ""
echo "  2. 导出模型"
echo "     python3 tools/export_model.py onnx --model $MODEL --checkpoint checkpoints/$MODEL/best.pth"
echo ""
echo "  3. TensorRT 优化"
echo "     python3 tools/export_model.py trt --onnx checkpoints/$MODEL/model.onnx --precision fp16"
echo ""
