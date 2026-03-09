@echo off
REM ============================================
REM 人脸识别端到端测试框架 - 环境配置脚本 (Windows)
REM ============================================

setlocal enabledelayedexpansion

REM 颜色定义（需要 Windows 10+ 支持 ANSI）
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
  set "DEL=%%a"
  set "COLOR_BLUE=%%b[34m"
  set "COLOR_GREEN=%%b[32m"
  set "COLOR_YELLOW=%%b[33m"
  set "COLOR_RED=%%b[31m"
  set "COLOR_RESET=%%b[0m"
)

REM 日志函数
:log_info
echo %COLOR_BLUE%[INFO]%COLOR_RESET% %1
goto :eof

:log_success
echo %COLOR_GREEN%[SUCCESS]%COLOR_RESET% %1
goto :eof

:log_warning
echo %COLOR_YELLOW%[WARNING]%COLOR_RESET% %1
goto :eof

:log_error
echo %COLOR_RED%[ERROR]%COLOR_RESET% %1
goto :eof

REM 获取脚本所在目录
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."

cd /d "%PROJECT_ROOT%"

echo ============================================
echo   人脸识别端到端测试框架 - 环境配置 (Windows)
echo ============================================
echo.

REM 检查 Python 版本
:check_python
call :log_info "检查 Python 版本..."
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "未找到 Python，请先安装 Python 3.9+"
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
call :log_success "Python 版本：%PYTHON_VERSION%"

REM 检查虚拟环境
:check_venv
call :log_info "检查虚拟环境..."
if exist "venv\Scripts\activate.bat" (
    call :log_warning "虚拟环境已存在，将重新创建"
    rmdir /s /q venv
)

REM 创建虚拟环境
:create_venv
call :log_info "创建 Python 虚拟环境..."
python -m venv venv
if errorlevel 1 (
    call :log_error "创建虚拟环境失败"
    exit /b 1
)
call :log_success "虚拟环境创建完成"

REM 激活虚拟环境
:activate_venv
call :log_info "激活虚拟环境..."
call venv\Scripts\activate.bat
call :log_success "虚拟环境已激活"

REM 升级 pip
:upgrade_pip
call :log_info "升级 pip..."
python -m pip install --upgrade pip setuptools wheel --quiet
call :log_success "pip 升级完成"

REM 安装 PyTorch
:install_pytorch
call :log_info "安装 PyTorch..."
REM 默认安装 CPU 版本，如有 GPU 可手动更改
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
call :log_success "PyTorch 安装完成"

REM 安装测试依赖
:install_test_deps
call :log_info "安装测试框架依赖..."
pip install -r tests\benchmarks\requirements.txt --quiet
if errorlevel 1 (
    call :log_warning "部分依赖安装失败，将继续..."
)
call :log_success "测试依赖安装完成"

REM 安装项目依赖
:install_project_deps
call :log_info "安装项目依赖..."
pip install -e . --quiet
call :log_success "项目依赖安装完成"

REM 创建必要目录
:create_directories
call :log_info "创建必要目录..."
if not exist "tests\benchmarks\datasets" mkdir tests\benchmarks\datasets
if not exist "tests\benchmarks\results" mkdir tests\benchmarks\results
if not exist "tests\benchmarks\reports" mkdir tests\benchmarks\reports
if not exist "tests\benchmarks\logs" mkdir tests\benchmarks\logs
if not exist "tests\benchmarks\checkpoints" mkdir tests\benchmarks\checkpoints
if not exist "datasets\lfw" mkdir datasets\lfw
if not exist "datasets\cfp_fp" mkdir datasets\cfp_fp
if not exist "datasets\agedb" mkdir datasets\agedb
if not exist "datasets\rfw" mkdir datasets\rfw
call :log_success "目录创建完成"

REM 验证安装
:verify_installation
call :log_info "验证安装..."
python -c "import torch; import cv2; import numpy as np; print(f'PyTorch: {torch.__version__}'); print(f'OpenCV: {cv2.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
call :log_success "安装验证完成"

echo.
echo ============================================
call :log_success "环境配置完成！"
echo ============================================
echo.
echo 使用方法:
echo   call venv\Scripts\activate.bat
echo   python tests\benchmarks\run_test.py --config tests\benchmarks\configs\default_config.yaml
echo.

pause
