#!/bin/bash

# ============================================
# DDFD-FaceRec 训练启动脚本
# ============================================

set -e

echo "========================================"
echo "DDFD-FaceRec 训练启动"
echo "========================================"

# 默认配置
DATASET_DIR="${DATASET_DIR:-datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
CONFIG_DIR="configs"
GPUS="${GPUS:-0}"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 帮助信息
show_help() {
    echo ""
    echo "用法：$0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model       模型类型 (detection|recognition)"
    echo "  --dataset     数据集路径 (默认：datasets)"
    echo "  --output      输出路径 (默认：checkpoints)"
    echo "  --gpus        GPU IDs (默认：0)"
    echo "  --batch-size  批次大小 (默认：32)"
    echo "  --epochs      训练轮数 (默认：100)"
    echo "  --resume      恢复训练的检查点路径"
    echo "  --help        显示帮助信息"
    echo ""
    echo "示例:"
    echo "  # 训练检测模型"
    echo "  $0 --model detection --dataset datasets/widerface"
    echo ""
    echo "  # 训练识别模型 (多 GPU)"
    echo "  $0 --model recognition --gpus 0,1,2,3 --batch-size 64"
    echo ""
    echo "  # 恢复训练"
    echo "  $0 --model detection --resume checkpoints/detection/epoch_50.pth"
    echo ""
}

# 解析参数
MODEL=""
RESUME=""
BATCH_SIZE=32
EPOCHS=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项：$1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查模型类型
if [ -z "$MODEL" ]; then
    echo -e "${RED}错误：请指定模型类型 (--model detection|recognition)${NC}"
    show_help
    exit 1
fi

# 检查数据集
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${YELLOW}警告：数据集目录不存在：$DATASET_DIR${NC}"
    echo "请先运行：python tools/prepare_datasets.py --dataset lfw"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR/$MODEL"

# 设置环境变量
export CUDA_VISIBLE_DEVICES="$GPUS"
echo -e "\n${YELLOW}使用 GPU: $GPUS${NC}"

# 检测模型训练
if [ "$MODEL" = "detection" ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}训练检测模型 (DKGA-Det)${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    CMD="python tools/train_detection.py \
        --config $CONFIG_DIR/detection/train.yaml \
        --data-dir $DATASET_DIR \
        --output-dir $OUTPUT_DIR/detection"
    
    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi
    
    echo -e "\n${YELLOW}执行命令:${NC}"
    echo "$CMD"
    echo ""
    
    eval $CMD

# 识别模型训练
elif [ "$MODEL" = "recognition" ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}训练识别模型 (DDFD-Rec)${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    CMD="python tools/train_recognition.py \
        --config $CONFIG_DIR/recognition/train.yaml \
        --data-dir $DATASET_DIR \
        --output-dir $OUTPUT_DIR/recognition"
    
    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi
    
    echo -e "\n${YELLOW}执行命令:${NC}"
    echo "$CMD"
    echo ""
    
    eval $CMD

else
    echo -e "${RED}错误：未知的模型类型 '$MODEL'${NC}"
    show_help
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}训练完成!${NC}"
echo -e "${GREEN}========================================${NC}"
