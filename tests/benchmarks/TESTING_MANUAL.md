# 人脸识别模型权威测试执行手册

## 📋 目录

1. [概述](#概述)
2. [测试环境搭建](#测试环境搭建)
3. [数据集准备](#数据集准备)
4. [模型适配](#模型适配)
5. [测试执行](#测试执行)
6. [结果分析](#结果分析)
7. [故障排查](#故障排查)
8. [附录](#附录)

---

## 概述

### 测试框架功能

本测试框架提供端到端的自动化测试方案，覆盖：

| 测试类型 | 测试内容 | 核心指标 |
|---------|---------|---------|
| **FRTE** | 1:1 验证、1:N 识别 | FNMR@FMR、Top-N 准确率 |
| **FIVE** | 视频流人脸检测 | FPS、跟踪准确率 |
| **FMD** | 变形人脸检测 | AP、FPR |
| **学术基准** | LFW、CFP-FP、AgeDB 等 | 准确率、AUC、EER |

### 测试流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 环境配置    │ -> │ 数据集下载  │ -> │ 模型加载    │ -> │ 测试执行    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│ 报告生成    │ <- │ 结果分析    │ <- │ 指标计算    │ <------┘
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 测试环境搭建

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| **操作系统** | Windows 10 / Linux Ubuntu 18.04 | Linux Ubuntu 20.04+ |
| **CPU** | 4 核心 | 8 核心+ |
| **内存** | 16 GB | 32 GB+ |
| **GPU** | 8 GB 显存 | RTX 3090/4090 (24GB) |
| **存储** | 50 GB 可用空间 | 200 GB+ SSD |

### 快速开始

#### Windows

```bash
# 进入测试框架目录
cd tests/benchmarks

# 运行环境配置脚本
setup_env.bat

# 激活虚拟环境
call ..\..\venv\Scripts\activate
```

#### Linux/Mac

```bash
# 进入测试框架目录
cd tests/benchmarks

# 运行环境配置脚本
chmod +x setup_env.sh
./setup_env.sh

# 激活虚拟环境
source ../../venv/bin/activate
```

### 手动安装依赖

```bash
# 安装 PyTorch (GPU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装测试框架依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 数据集准备

### 支持的数据集

| 数据集 | 用途 | 自动下载 | 需要授权 |
|-------|------|---------|---------|
| LFW | 1:1 验证 | ✅ | ❌ |
| CFP-FP | 跨姿态识别 | ❌ | ✅ |
| AgeDB-30 | 跨年龄识别 | ❌ | ✅ |
| RFW | 跨种族公平性 | ✅ | ❌ |
| WIDER Face | 人脸检测 | ✅ | ❌ |
| IJB-C | 遮挡/光照鲁棒性 | ❌ | ✅ |
| MegaFace | 百万级 1:N | ❌ | ✅ |

### 自动下载数据集

```bash
# 下载 LFW 数据集
python tests/benchmarks/datasets/dataset_download.py --download lfw lfw_pairs

# 下载 WIDER Face
python tests/benchmarks/datasets/dataset_download.py --download wider_face

# 列出所有支持的数据集
python tests/benchmarks/datasets/dataset_download.py --list-datasets
```

### 手动下载数据集

#### LFW

1. 访问：http://vis-www.cs.umass.edu/lfw/
2. 下载 `lfw-deepfunneled.tgz`
3. 解压到 `datasets/lfw/`

#### CFP-FP

1. 访问：https://www.cfp-biometrics.org/
2. 注册并申请访问权限
3. 下载数据集到 `datasets/cfp_fp/`

### 数据集目录结构

```
datasets/
├── lfw/
│   ├── lfw/
│   │   ├── Aaron_Eckhart/
│   │   │   ├── Aaron_Eckhart_0001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── pairs.txt
├── cfp_fp/
│   ├── frontal/
│   └── profile/
├── agedb/
│   └── AgeDB/
└── widerface/
    └── WIDER_face/
```

---

## 模型适配

### 接口定义

测试框架定义了标准接口，用户需要实现这些接口来适配自研模型。

#### 人脸检测模型接口

```python
from tests.benchmarks.run_test import DetectorInterface

class MyDetector(DetectorInterface):
    def load_model(self, checkpoint_path: str):
        """加载模型权重"""
        # TODO: 实现模型加载
        self.model = YourModel()
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理输入图像"""
        # TODO: 实现预处理逻辑
        pass
        
    def inference(self, inputs: torch.Tensor) -> Dict:
        """模型推理"""
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
        
    def postprocess(self, outputs: Dict) -> List[Dict]:
        """后处理输出"""
        # TODO: 实现 NMS、坐标转换等
        return [
            {
                'bbox': [x1, y1, x2, y2],
                'score': confidence,
                'landmarks': [[x1,y1], [x2,y2], ...]  # 5 个关键点
            },
            ...
        ]
```

#### 特征提取模型接口

```python
from tests.benchmarks.run_test import RecognizerInterface

class MyRecognizer(RecognizerInterface):
    def load_model(self, checkpoint_path: str):
        """加载模型权重"""
        self.model = YourRecognitionModel()
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理人脸图像"""
        pass
        
    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        with torch.no_grad():
            feature = self.model(inputs)
        return feature
        
    def postprocess(self, outputs: np.ndarray) -> np.ndarray:
        """L2 归一化"""
        return outputs / (np.linalg.norm(outputs) + 1e-10)
```

### 适配示例

参考 `tests/benchmarks/run_test.py` 中的 `CustomDetector` 和 `CustomRecognizer` 类。

---

## 测试执行

### 配置文件

编辑 `tests/benchmarks/configs/default_config.yaml`：

```yaml
model:
  detector:
    checkpoint: "checkpoints/detection/best.pth"
    input_size: [640, 640]
    score_thresh: 0.5
    
  recognizer:
    checkpoint: "checkpoints/recognition/best.pth"
    input_size: [112, 112]
    feature_dim: 512
    
  matcher:
    threshold: 0.6
    metric: "cosine"

datasets:
  academic:
    lfw:
      enabled: true
      data_dir: "datasets/lfw"
```

### 运行测试

#### 完整测试

```bash
# 使用默认配置运行所有测试
python tests/benchmarks/run_test.py --config tests/benchmarks/configs/default_config.yaml
```

#### 单独测试

```bash
# 只测试人脸检测
python tests/benchmarks/run_test.py --test detection --dataset widerface

# 只测试人脸识别
python tests/benchmarks/run_test.py --test recognition --dataset lfw
```

#### 自定义模型路径

```bash
python tests/benchmarks/run_test.py \
  --detector_ckpt checkpoints/detection/my_model.pth \
  --recognizer_ckpt checkpoints/recognition/my_model.pth
```

### 测试进度监控

测试过程中会实时输出：

```
12:30:45 | INFO     | 开始人脸识别模块测试
12:30:45 | INFO     | 加载 LFW pairs: 6000 对
Extracting features: 100%|████████| 6000/6000 [02:15<00:00, 44.3it/s]
12:33:00 | INFO     | AUC: 0.9876
12:33:00 | INFO     | EER: 0.0456
```

---

## 结果分析

### 输出文件

测试完成后，以下文件会生成：

```
tests/benchmarks/
├── results/
│   └── test_results_YYYYMMDD_HHMMSS.json    # 原始结果
├── reports/
│   ├── report_YYYYMMDD_HHMMSS.html          # HTML 报告
│   ├── report_YYYYMMDD_HHMMSS.md            # Markdown 报告
│   └── report_YYYYMMDD_HHMMSS.json          # JSON 报告
└── logs/
    └── test_YYYYMMDD_HHMMSS.log             # 详细日志
```

### 核心指标解读

#### FRTE 指标

| 指标 | 含义 | 优秀标准 |
|-----|------|---------|
| **AUC** | ROC 曲线下面积 | > 0.99 |
| **EER** | 等错误率 | < 0.05 |
| **FNMR@FMR=10⁻⁴** | FMR=0.01% 时的 FNMR | < 0.1 |
| **FNMR@FMR=10⁻⁶** | FMR=0.0001% 时的 FNMR | < 0.3 |

#### 检测指标

| 指标 | 含义 | 优秀标准 |
|-----|------|---------|
| **mAP** | 平均精度均值 | > 0.95 |
| **AP@0.5** | IoU=0.5 时的 AP | > 0.98 |
| **Recall** | 召回率 | > 0.95 |

### 与 SOTA 对比

配置文件中可设置对比模型：

```yaml
report:
  comparison_models:
    - name: "ArcFace"
      lfw_accuracy: 0.9983
      cfp_fp_accuracy: 0.9958
    - name: "FaceNet"
      lfw_accuracy: 0.9963
```

### 生成报告

```bash
# 从 JSON 结果生成报告
python tests/benchmarks/reports/report_generator.py \
  --results tests/benchmarks/results/test_results_*.json \
  --format all
```

---

## 故障排查

### 常见问题

#### 1. CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```yaml
# 减小 batch_size
test:
  batch_size: 16  # 改为 8 或更小
```

#### 2. 数据集下载失败

**症状**: `ConnectionError` 或 `TimeoutError`

**解决方案**:
- 使用镜像源
- 手动下载后放到指定目录

#### 3. 模型加载失败

**症状**: `KeyError` 或 `SizeMismatchError`

**解决方案**:
- 检查 checkpoint 路径
- 确认模型结构匹配

### 日志分析

查看详细日志：

```bash
# 查看最近的日志
tail -f tests/benchmarks/logs/test_*.log
```

---

## 附录

### A. 命令行参数

```
usage: run_test.py [-h] [--config CONFIG] [--test {detection,recognition,all}]
                   [--dataset DATASET] [--detector_ckpt DETECTOR_CKPT]
                   [--recognizer_ckpt RECOGNIZER_CKPT]

optional arguments:
  -h, --help            显示帮助信息
  --config CONFIG       配置文件路径
  --test TYPE           测试类型 (detection/recognition/all)
  --dataset DATASET     测试数据集
  --detector_ckpt PATH  检测模型权重路径
  --recognizer_ckpt PATH 识别模型权重路径
```

### B. 配置模板

完整配置参考 `tests/benchmarks/configs/default_config.yaml`

### C. 联系支持

如有问题，请提交 Issue 或联系开发团队。

---

*文档版本：v1.0*
*最后更新：2026 年 3 月*
