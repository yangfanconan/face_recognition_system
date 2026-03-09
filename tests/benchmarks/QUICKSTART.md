# 人脸识别端到端测试框架 - 快速开始

## 🚀 5 分钟快速上手

### 步骤 1：环境配置

```bash
# Windows
cd tests/benchmarks
setup_env.bat

# Linux/Mac
cd tests/benchmarks
chmod +x setup_env.sh
./setup_env.sh
```

### 步骤 2：下载测试数据集

```bash
# 下载 LFW 数据集（用于识别测试）
python datasets/dataset_download.py --download lfw lfw_pairs

# 下载 WIDER Face（用于检测测试）
python datasets/dataset_download.py --download wider_face
```

### 步骤 3：配置模型路径

编辑 `configs/default_config.yaml`：

```yaml
model:
  detector:
    checkpoint: "checkpoints/detection/best.pth"  # 你的检测模型路径
  recognizer:
    checkpoint: "checkpoints/recognition/best.pth"  # 你的识别模型路径
```

### 步骤 4：运行测试

```bash
# 运行完整测试
python run_test.py --config configs/default_config.yaml

# 或只运行识别测试
python run_test.py --test recognition --dataset lfw
```

### 步骤 5：查看报告

测试完成后，报告生成在：

```
tests/benchmarks/reports/
├── report_YYYYMMDD_HHMMSS.html   # HTML 报告（推荐）
├── report_YYYYMMDD_HHMMSS.md     # Markdown 报告
└── report_YYYYMMDD_HHMMSS.json   # JSON 结果
```

---

## 📊 预期输出

```
============================================
人脸识别端到端自动化测试框架
============================================

环境信息:
  Python: 3.9.0
  PyTorch: 2.1.0
  CUDA: True
  GPU: NVIDIA GeForce RTX 4090

========================================
开始人脸识别模块测试
========================================

加载 LFW pairs: 6000 对
Extracting features: 100%|████████| 6000/6000 [02:15<00:00, 44.3it/s]

测试结果:
  AUC: 0.9876
  EER: 0.0456
  FNMR@FMR=1e-4: 0.1234
  FNMR@FMR=1e-6: 0.2345

测试结果已保存：tests/benchmarks/results/test_results_20260309_120000.json
HTML 报告已生成：tests/benchmarks/reports/report_20260309_120000.html

测试完成!
```

---

## 🔧 自定义测试

### 测试特定数据集

```bash
# CFP-FP（跨姿态）
python run_test.py --test recognition --dataset cfp_fp

# AgeDB（跨年龄）
python run_test.py --test recognition --dataset agedb

# RFW（跨种族）
python run_test.py --test recognition --dataset rfw
```

### 调整测试参数

```yaml
# configs/default_config.yaml
test:
  batch_size: 32          # 批大小
  num_workers: 4          # 数据加载线程数
  log_interval: 100       # 日志打印间隔
  
metrics:
  recognition:
    - "accuracy"
    - "fnmr_at_fmr"
    - "top1_accuracy"
    - "top5_accuracy"
```

---

## 📁 项目结构

```
tests/benchmarks/
├── configs/                  # 配置文件
│   └── default_config.yaml
├── datasets/                 # 数据集处理
│   ├── dataset_download.py
│   └── __init__.py
├── metrics/                  # 指标计算
│   ├── nist_metrics.py
│   └── __init__.py
├── reports/                  # 报告生成
│   ├── report_generator.py
│   └── __init__.py
├── results/                  # 测试结果（自动生成）
├── logs/                     # 日志文件（自动生成）
├── __init__.py
├── run_test.py              # 主测试入口
├── setup_env.bat            # Windows 环境配置
├── setup_env.sh             # Linux 环境配置
├── TESTING_MANUAL.md        # 测试执行手册
├── ANALYSIS_GUIDE.md        # 结果分析指南
└── requirements.txt         # 依赖列表
```

---

## 📖 详细文档

- **测试执行手册**: `TESTING_MANUAL.md`
- **分析指南**: `ANALYSIS_GUIDE.md`
- **配置示例**: `configs/default_config.yaml`

---

## ❓ 常见问题

**Q: 测试需要多长时间？**
A: LFW 完整测试约 5-10 分钟（RTX 4090），取决于模型大小。

**Q: 如何适配自研模型？**
A: 继承 `DetectorInterface` 和 `RecognizerInterface` 类，实现相应方法。

**Q: 测试失败怎么办？**
A: 查看 `logs/` 目录下的详细日志，参考 `TESTING_MANUAL.md` 故障排查章节。

---

*快速开始版本：v1.0*
*最后更新：2026 年 3 月*
