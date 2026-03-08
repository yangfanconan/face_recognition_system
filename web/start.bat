@echo off
echo ============================================
echo DDFD-FaceRec Web 演示启动脚本
echo ============================================
echo.

cd /d "%~dp0"

echo 检查虚拟环境...
if not exist "..\venv\Scripts\python.exe" (
    echo 错误：虚拟环境不存在！
    echo 请先运行：cd .. ^&^& python -m venv venv
    pause
    exit /b 1
)

echo 安装 Web 依赖...
call ..\venv\Scripts\pip install -r requirements.txt

echo.
echo 启动 Web 服务...
echo 访问地址：http://localhost:7860
echo 按 Ctrl+C 停止服务
echo.

call ..\venv\Scripts\python app.py

pause
