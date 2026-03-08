@echo off
REM ============================================
REM DDFD-FaceRec Windows 环境配置脚本
REM ============================================

echo ========================================
echo DDFD-FaceRec 环境配置 (Windows)
echo ========================================

REM 检查 Python
echo.
echo 检查 Python 版本...
python --version

REM 创建虚拟环境
echo.
echo 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    echo 虚拟环境创建成功
) else (
    echo 虚拟环境已存在
)

REM 激活虚拟环境
echo.
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 升级 pip
echo.
echo 升级 pip...
python -m pip install --upgrade pip

REM 安装 PyTorch (CUDA 12.0)
echo.
echo 安装 PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

REM 安装项目依赖
echo.
echo 安装项目依赖...
pip install -r requirements.txt

REM 安装开发依赖
echo.
echo 安装开发依赖...
pip install pytest pytest-cov black flake8 mypy

REM 安装项目
echo.
echo 安装项目...
pip install -e .

REM 验证安装
echo.
echo 验证安装...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA 可用：{torch.cuda.is_available()}')"

REM 创建目录
echo.
echo 创建必要目录...
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "datasets" mkdir datasets
if not exist "storage" mkdir storage

echo.
echo ========================================
echo 环境配置完成!
echo ========================================
echo.
echo 使用指南:
echo 1. 激活环境：venv\Scripts\activate
echo 2. 下载数据：python tools\download_datasets.py --dataset lfw
echo 3. 训练模型：python tools\train_detection.py --config configs\detection\train.yaml
echo 4. 启动 API: python -m api.main
echo.

pause
