"""
训练配置检查工具

检查训练环境配置是否正确
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """检查 Python 版本"""
    import sys
    version = sys.version_info
    print(f"Python 版本：{version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("  ⚠️  警告：Python 版本低于 3.10，可能不兼容")
        return False
    print("  ✅ Python 版本符合要求 (>=3.10)")
    return True


def check_torch():
    """检查 PyTorch 安装"""
    try:
        import torch
        print(f"\nPyTorch 版本：{torch.__version__}")
        print(f"CUDA 可用：{torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 版本：{torch.version.cuda}")
            print(f"GPU 数量：{torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print("  ✅ GPU 可用")
        else:
            print("  ⚠️  警告：CUDA 不可用，将使用 CPU 训练")
        
        return True
    except ImportError:
        print("\n❌ PyTorch 未安装")
        print("  安装命令：pip install torch torchvision torchaudio")
        return False


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'torchvision',
        'numpy',
        'cv2',
        'yaml',
        'tqdm',
        'scipy',
        'sklearn',
    ]
    
    print("\n检查依赖包:")
    all_installed = True
    
    for package in required_packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {package}: {version}")
        except ImportError:
            print(f"  ❌ {package}: 未安装")
            all_installed = False
    
    # 可选依赖
    optional_packages = [
        ('tensorboard', 'TensorBoard'),
        ('wandb', 'Weights & Biases'),
        ('hnswlib', 'HNSW 索引'),
        ('onnx', 'ONNX'),
        ('onnxruntime', 'ONNX Runtime'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
    ]
    
    print("\n可选依赖:")
    for package, name in optional_packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {name}: {version}")
        except ImportError:
            print(f"  ⚪ {name}: 未安装 (可选)")
    
    return all_installed


def check_directories():
    """检查目录结构"""
    print("\n检查目录结构:")
    
    required_dirs = [
        'configs',
        'models',
        'data',
        'tools',
        'inference',
        'checkpoints',
        'logs',
        'datasets',
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(__file__).parent.parent / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (缺失)")
            all_exist = False
    
    return all_exist


def check_config_files():
    """检查配置文件"""
    print("\n检查配置文件:")
    
    config_files = [
        'configs/default.yaml',
        'configs/detection/train.yaml',
        'configs/recognition/train.yaml',
        'configs/deployment/infer.yaml',
    ]
    
    all_exist = True
    for config_file in config_files:
        config_path = Path(__file__).parent.parent / config_file
        if config_path.exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file} (缺失)")
            all_exist = False
    
    return all_exist


def check_gpu_memory():
    """检查 GPU 显存"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("\n⚠️  CUDA 不可用，跳过 GPU 显存检查")
            return True
        
        print("\nGPU 显存检查:")
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            
            print(f"  GPU {i}:")
            print(f"    总显存：{total:.2f} GB")
            print(f"    已分配：{allocated:.2f} GB")
            print(f"    已预留：{reserved:.2f} GB")
            print(f"    可用：{total - allocated:.2f} GB")
            
            if total < 8:
                print(f"    ⚠️  警告：显存小于 8GB，可能需要减小 batch size")
            else:
                print(f"    ✅ 显存充足")
        
        return True
    except Exception as e:
        print(f"\n❌ GPU 显存检查失败：{e}")
        return False


def check_disk_space():
    """检查磁盘空间"""
    print("\n磁盘空间检查:")
    
    import shutil
    
    root_path = Path(__file__).parent.parent
    total, used, free = shutil.disk_usage(root_path)
    
    free_gb = free / 1024**3
    
    print(f"  总空间：{total / 1024**3:.2f} GB")
    print(f"  已使用：{used / 1024**3:.2f} GB")
    print(f"  可用空间：{free_gb:.2f} GB")
    
    if free_gb < 10:
        print("  ⚠️  警告：可用空间小于 10GB")
        print("     训练数据集可能需要 50-100GB 空间")
    elif free_gb < 50:
        print("  ⚠️  警告：可用空间小于 50GB")
        print("     可能不足以存储完整数据集")
    else:
        print("  ✅ 磁盘空间充足")
    
    return True


def run_all_checks():
    """运行所有检查"""
    print("=" * 60)
    print("DDFD-FaceRec 训练环境检查")
    print("=" * 60)
    
    results = []
    
    results.append(("Python 版本", check_python_version()))
    results.append(("PyTorch", check_torch()))
    results.append(("依赖包", check_dependencies()))
    results.append(("目录结构", check_directories()))
    results.append(("配置文件", check_config_files()))
    results.append(("GPU 显存", check_gpu_memory()))
    results.append(("磁盘空间", check_disk_space()))
    
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有检查通过！环境配置正确。")
    else:
        print("⚠️  部分检查未通过，请根据提示修复。")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
