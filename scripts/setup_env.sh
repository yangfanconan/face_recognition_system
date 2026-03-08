#!/bin/bash

# ============================================
# DDFD-FaceRec 环境配置脚本
# ============================================

set -e

echo "========================================"
echo "DDFD-FaceRec 环境配置"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python 版本
echo -e "\n${YELLOW}检查 Python 版本...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本：$python_version"

# 创建虚拟环境
echo -e "\n${YELLOW}创建虚拟环境...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}虚拟环境创建成功${NC}"
else
    echo -e "${YELLOW}虚拟环境已存在${NC}"
fi

# 激活虚拟环境
echo -e "\n${YELLOW}激活虚拟环境...${NC}"
source venv/bin/activate

# 升级 pip
echo -e "\n${YELLOW}升级 pip...${NC}"
pip install --upgrade pip

# 安装基础依赖
echo -e "\n${YELLOW}安装基础依赖...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

# 安装项目依赖
echo -e "\n${YELLOW}安装项目依赖...${NC}"
pip install -r requirements.txt

# 安装开发依赖
echo -e "\n${YELLOW}安装开发依赖...${NC}"
pip install pytest pytest-cov black flake8 mypy

# 安装项目
echo -e "\n${YELLOW}安装项目...${NC}"
pip install -e .

# 验证安装
echo -e "\n${YELLOW}验证安装...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

# 检查 CUDA
echo -e "\n${YELLOW}检查 CUDA 可用性...${NC}"
python -c "import torch; print(f'CUDA 可用：{torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA 版本：{torch.version.cuda}' if torch.cuda.is_available() else 'N/A')"
python -c "import torch; print(f'GPU 数量：{torch.cuda.device_count()}' if torch.cuda.is_available() else 'N/A')"

# 创建必要目录
echo -e "\n${YELLOW}创建必要目录...${NC}"
mkdir -p checkpoints
mkdir -p logs
mkdir -p datasets
mkdir -p storage

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}环境配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}使用指南:${NC}"
echo "1. 激活环境：source venv/bin/activate"
echo "2. 下载数据：python tools/download_datasets.py --dataset lfw"
echo "3. 训练模型：python tools/train_detection.py --config configs/detection/train.yaml"
echo "4. 启动 API: python -m api.main"
echo ""
