#!/bin/bash
# ============================================
# 人脸识别端到端测试框架 - 环境配置脚本 (Linux/Mac)
# ============================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Python 版本
check_python_version() {
    log_info "检查 Python 版本..."
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    REQUIRED_VERSION="3.9"
    
    if [[ $(python3 -c "import sys; print(sys.version_info >= (3, 9))") == "True" ]]; then
        log_success "Python 版本：$PYTHON_VERSION (满足要求 >= 3.9)"
    else
        log_error "Python 版本：$PYTHON_VERSION (需要 >= 3.9)"
        exit 1
    fi
}

# 检查 GPU 和 CUDA
check_gpu() {
    log_info "检查 GPU 和 CUDA..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        log_success "检测到 NVIDIA GPU: $GPU_INFO"
    else
        log_warning "未检测到 NVIDIA GPU，将使用 CPU 模式"
    fi
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',')
        log_success "CUDA 版本：$CUDA_VERSION"
    else
        log_warning "未检测到 CUDA Toolkit"
    fi
}

# 创建虚拟环境
create_venv() {
    log_info "创建 Python 虚拟环境..."
    
    if [ -d "venv" ]; then
        log_warning "虚拟环境已存在，将重新创建"
        rm -rf venv
    fi
    
    python3 -m venv venv
    log_success "虚拟环境创建完成"
}

# 激活虚拟环境
activate_venv() {
    log_info "激活虚拟环境..."
    source venv/bin/activate
    log_success "虚拟环境已激活"
}

# 升级 pip
upgrade_pip() {
    log_info "升级 pip..."
    pip install --upgrade pip setuptools wheel
    log_success "pip 升级完成"
}

# 安装 PyTorch
install_pytorch() {
    log_info "安装 PyTorch..."
    
    # 检测 CUDA 版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d'.' -f1,2)
        log_info "检测到 CUDA $CUDA_VERSION，安装 GPU 版本 PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        log_warning "未检测到 CUDA，安装 CPU 版本 PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    log_success "PyTorch 安装完成"
}

# 安装测试依赖
install_test_deps() {
    log_info "安装测试框架依赖..."
    pip install -r tests/benchmarks/requirements.txt
    log_success "测试依赖安装完成"
}

# 安装项目依赖
install_project_deps() {
    log_info "安装项目依赖..."
    pip install -e .
    log_success "项目依赖安装完成"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    mkdir -p tests/benchmarks/{datasets,results,reports,logs,checkpoints}
    mkdir -p datasets/{lfw,cfp_fp,agedb,rfw,widerface}
    log_success "目录创建完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    python3 -c "
import torch
import cv2
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
"
    
    log_success "安装验证完成"
}

# 主函数
main() {
    echo "============================================"
    echo "  人脸识别端到端测试框架 - 环境配置"
    echo "============================================"
    echo ""
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/../../"
    
    check_python_version
    check_gpu
    create_venv
    activate_venv
    upgrade_pip
    install_pytorch
    install_test_deps
    install_project_deps
    create_directories
    verify_installation
    
    echo ""
    echo "============================================"
    log_success "环境配置完成！"
    echo "============================================"
    echo ""
    echo "使用方法:"
    echo "  source venv/bin/activate"
    echo "  python tests/benchmarks/run_test.py --config tests/benchmarks/configs/default_config.yaml"
    echo ""
}

# 执行主函数
main "$@"
