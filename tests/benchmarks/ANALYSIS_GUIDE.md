# 人脸识别测试结果分析指南

## 📊 指标解读与性能对标

### 一、核心指标详解

#### 1. FRTE 指标（NIST 标准）

##### FNMR@FMR（False Non-Match Rate @ False Match Rate）

**定义**: 在指定误匹配率（FMR）下的非匹配率（FNMR）

**计算逻辑**:
```python
# 对于给定的阈值 threshold
FMR = FP / (FP + TN)  # 负样本被错误接受的比例
FNMR = FN / (FN + TP)  # 正样本被错误拒绝的比例

# FNMR@FMR=10⁻⁴ 表示当 FMR=0.01% 时的 FNMR 值
```

**行业标准**:
| 应用场景 | FMR 目标 | FNMR 要求 |
|---------|---------|----------|
| 手机解锁 | 10⁻³ | < 0.05 |
| 金融支付 | 10⁻⁴ | < 0.1 |
| 安防门禁 | 10⁻⁵ | < 0.2 |
| 司法鉴定 | 10⁻⁶ | < 0.3 |

**分析建议**:
- FNMR 越低，用户体验越好（少被误拒）
- FMR 越低，安全性越高（少被误入）
- 两者是权衡关系，需根据场景选择

---

##### EER（Equal Error Rate）

**定义**: FMR = FNMR 时的错误率

**解读**:
- EER 越低，模型整体性能越好
- 是衡量模型综合能力的单一指标

**性能分级**:
| EER 范围 | 评级 | 适用场景 |
|---------|------|---------|
| < 0.01 | 优秀 | 金融级、司法级 |
| 0.01-0.05 | 良好 | 商业门禁、手机解锁 |
| 0.05-0.10 | 一般 | 一般安防 |
| > 0.10 | 较差 | 需优化 |

---

##### AUC（Area Under Curve）

**定义**: ROC 曲线下的面积

**解读**:
- AUC = 1.0：完美分类器
- AUC = 0.5：随机猜测
- AUC > 0.99：优秀模型

---

#### 2. 检测指标

##### mAP（mean Average Precision）

**COCO 标准**:
| mAP | 评级 |
|-----|------|
| > 0.75 | 优秀 |
| 0.50-0.75 | 良好 |
| < 0.50 | 需优化 |

**WIDER Face 标准**:
| 场景 | 优秀 mAP |
|-----|---------|
| Easy | > 0.95 |
| Medium | > 0.90 |
| Hard | > 0.80 |

---

### 二、性能对标分析

#### 与 SOTA 模型对比

```
┌─────────────────────────────────────────────────────────────┐
│ LFW 验证准确率对比                                           │
├─────────────────────────────────────────────────────────────┤
│ ArcFace    ████████████████████████████████████  99.83%     │
│ FaceNet    ██████████████████████████████████░░  99.63%     │
│ CosFace    ███████████████████████████████████░  99.73%     │
│ 自研模型   ████████████████████████████████░░░░  99.45%     │
└─────────────────────────────────────────────────────────────┘
```

#### 差距分析框架

**1. 数据层面**
- [ ] 训练数据量是否充足？
- [ ] 数据质量（清晰度、多样性）如何？
- [ ] 是否存在数据偏差？

**2. 模型层面**
- [ ] 模型容量是否足够？
- [ ] 损失函数选择是否合适？
- [ ] 是否存在过拟合/欠拟合？

**3. 训练策略**
- [ ] 学习率设置是否合理？
- [ ] 数据增强是否充分？
- [ ] 是否使用了预训练模型？

---

### 三、场景化分析

#### 1. 跨姿态性能（CFP-FP）

**分析维度**:
```python
# 按姿态角度分组分析
pose_ranges = [
    (0, 15),    # 正面
    (15, 30),   # 小侧脸
    (30, 45),   # 中侧脸
    (45, 90)    # 大侧脸
]

# 计算各姿态区间的准确率
for min_angle, max_angle in pose_ranges:
    accuracy = compute_accuracy(pose_range=(min_angle, max_angle))
    print(f"{min_angle}-{max_angle}°: {accuracy:.2%}")
```

**优化建议**:
- 大姿态性能下降 → 增加侧脸训练数据
- 使用 3D 人脸校正
- 引入姿态不变特征

---

#### 2. 跨年龄性能（AgeDB）

**年龄分组分析**:
| 年龄跨度 | 准确率 | 问题分析 |
|---------|-------|---------|
| 0-5 年 | 99.5% | 表现良好 |
| 5-10 年 | 97.8% | 轻微下降 |
| 10-20 年 | 94.2% | 明显下降 |
| 20+ 年 | 89.5% | 需优化 |

**优化建议**:
- 引入年龄不变损失
- 使用年龄条件化特征
- 增加跨年龄训练样本

---

#### 3. 跨种族公平性（RFW）

**公平性指标**:
```python
# 计算各族群的准确率差异
accuracies = {
    'Caucasian': 0.995,
    'Asian': 0.992,
    'Indian': 0.985,
    'African': 0.978
}

# 公平性差距
gap = max(accuracies.values()) - min(accuracies.values())
print(f"种族间准确率差距：{gap:.2%}")
```

**公平性标准**:
| 差距 | 评级 |
|-----|------|
| < 1% | 优秀 |
| 1-3% | 可接受 |
| > 3% | 需优化 |

---

### 四、异常样本分析

#### 1. 错误类型分类

```python
error_categories = {
    'pose': '姿态过大（>45°）',
    'occlusion': '遮挡（口罩、墨镜等）',
    'blur': '图像模糊',
    'lighting': '光照异常（过暗/过曝）',
    'resolution': '分辨率过低',
    'age': '年龄跨度过大',
    'expression': '表情夸张'
}
```

#### 2. 自动筛选错误样本

```python
def analyze_failures(results, threshold=0.6):
    """分析失败样本"""
    failures = []
    
    for pred in results:
        if pred['label'] == 1 and pred['score'] < threshold:
            # FN 错误（正样本被拒绝）
            failures.append({
                'type': 'FN',
                'sample': pred['sample'],
                'score': pred['score'],
                'possible_reason': categorize_error(pred['sample'])
            })
        elif pred['label'] == 0 and pred['score'] >= threshold:
            # FP 错误（负样本被接受）
            failures.append({
                'type': 'FP',
                'sample': pred['sample'],
                'score': pred['score'],
                'possible_reason': categorize_error(pred['sample'])
            })
            
    return failures
```

#### 3. 错误样本可视化

```python
def visualize_failures(failures, output_dir="failure_samples"):
    """可视化失败样本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 按错误类型分组
    by_type = defaultdict(list)
    for f in failures:
        by_type[f['possible_reason']].append(f)
        
    # 每种类型显示 10 个样本
    for error_type, samples in by_type.items():
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i, sample in enumerate(samples[:10]):
            ax = axes[i // 5, i % 5]
            img = cv2.imread(sample['sample']['image_path'])
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Score: {sample['score']:.3f}")
            ax.axis('off')
        plt.suptitle(f"{error_type} ({len(samples)} samples)")
        plt.savefig(f"{output_dir}/{error_type}.png")
```

---

### 五、优化建议生成

#### 基于指标的诊断树

```
AUC < 0.95?
├─ Yes → 模型容量不足或训练不充分
│  ├─ 增加模型层数/通道数
│  └─ 增加训练轮次
│
└─ No → 进入下一步

EER > 0.05?
├─ Yes → 阈值校准或特征区分度不够
│  ├─ 在验证集上重新校准阈值
│  └─ 使用更强的损失函数（ArcFace/CosFace）
│
└─ No → 进入下一步

FNMR@FMR=10⁻⁴ > 0.1?
├─ Yes → 高安全性场景性能不足
│  ├─ 增加难样本挖掘
│  └─ 调整损失函数的边界参数
│
└─ No → 性能良好
```

#### 场景化优化建议

**金融级 1:1 验证**:
- 目标：FNMR@FMR=10⁻⁴ < 0.05
- 建议:
  1. 使用 ArcFace 损失，margin=0.5
  2. 特征维度 >= 512
  3. 在验证集上精细校准阈值

**安防 1:N 检索**:
- 目标：Top-1 准确率 > 95% @ N=10000
- 建议:
  1. 使用 HNSW 索引加速检索
  2. 特征归一化后使用余弦相似度
  3. 考虑特征融合（多模型集成）

**移动端低功耗**:
- 目标：推理时间 < 100ms
- 建议:
  1. 模型蒸馏（Teacher-Student）
  2. 量化（FP16/INT8）
  3. 使用轻量级 backbone（MobileFaceNet）

---

### 六、报告模板

#### 测试结果摘要

```markdown
## 测试结果摘要

### 整体性能
- **验证 AUC**: 0.9876 (vs ArcFace 0.9983, -1.07%)
- **等错误率**: 0.0456 (vs ArcFace 0.0321, +42%)
- **FNMR@FMR=10⁻⁴**: 0.1234 (目标 < 0.1, ❌ 未达标)

### 数据集表现
| 数据集 | 准确率 | 与 SOTA 差距 |
|-------|-------|-------------|
| LFW | 99.45% | -0.38% |
| CFP-FP | 97.82% | -1.76% |
| AgeDB-30 | 96.54% | -0.89% |

### 性能瓶颈
1. 大姿态场景（CFP-FP）性能下降明显
2. 跨年龄识别（AgeDB）有待提升
3. 高安全性要求下（FMR=10⁻⁶）FNMR 偏高

### 优化优先级
1. 🔴 高：增加侧脸训练数据
2. 🟡 中：调整损失函数参数
3. 🟢 低：模型容量扩展
```

---

### 七、可视化模板

#### ROC 曲线对比

```python
def plot_roc_comparison(results_dict, save_path="roc_comparison.png"):
    """多模型 ROC 对比"""
    plt.figure(figsize=(10, 8))
    
    for name, data in results_dict.items():
        plt.plot(data['fmr'], data['fnmr'], label=name, linewidth=2)
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
```

#### 指标雷达图

```python
def plot_radar_chart(metrics_dict, save_path="radar_chart.png"):
    """多指标雷达图"""
    categories = ['LFW', 'CFP-FP', 'AgeDB', 'IJB-C', 'RFW']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for name, metrics in metrics_dict.items():
        values = [metrics.get(cat, 0) for cat in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.25)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(save_path, dpi=150)
```

---

*分析指南版本：v1.0*
*最后更新：2026 年 3 月*
