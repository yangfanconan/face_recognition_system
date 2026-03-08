# DDFD-FaceRec 模型修复指南

**创建日期**: 2026 年 3 月 7 日  
**优先级**: 高

---

## 🔧 已知问题

### 问题 1: 检测器通道数不匹配

**错误信息**:
```
Given groups=1, weight of size [128, 64, 3, 3], 
expected input[1, 128, 160, 160] to have 64 channels, 
but got 128 channels instead
```

**问题定位**: `models/detection/neck.py` - BiFPNLite

**原因分析**:
- P2 层处理时，输入通道数计算错误
- `in_channels[0]` 应该是 256 (P3), 但实际传入了 128

**修复方案**:

```python
# models/detection/neck.py - BiFPNLite.__init__
def __init__(
    self,
    in_channels: List[int] = [256, 512, 1024],  # P3, P4, P5
    out_channels: int = 256,
    num_layers: int = 2,
    use_p2: bool = True,
    attention: bool = True
):
    self.use_p2 = use_p2
    
    # P2 层处理 - 从 P3 上采样
    if use_p2:
        # in_channels[0] 是 P3 的通道数 (256)
        self.p2_conv = nn.Conv2d(in_channels[0], out_channels, 1, bias=False)
        # 修改后：P2 输出 out_channels, 所以后续应该用 out_channels
        in_channels_for_bifpn = [out_channels] + in_channels  # [256, 256, 512, 1024]
    else:
        in_channels_for_bifpn = in_channels
    
    # BiFPN 层 - 使用正确的通道数
    self.bifpn_layers = nn.ModuleList([
        BiFPNBlock(
            in_channels=in_channels_for_bifpn,  # 修复这里
            out_channels=out_channels,
            num_levels=len(in_channels_for_bifpn),
            attention_type="se" if attention else None
        )
        for _ in range(num_layers)
    ])
```

---

### 问题 2: 识别器维度不匹配

**错误信息**:
```
The size of tensor a (28) must match the size of tensor b (56) 
at non-singleton dimension 3
```

**问题定位**: `models/recognition/fusion.py` - FrequencyGatedFusion

**原因分析**:
- 空域分支输出 512 通道，频域分支输出 256 通道
- 融合时通道数不匹配

**修复方案**:

```python
# models/recognition/fusion.py - FrequencyGatedFusion
def __init__(
    self,
    channels: int,
    freq_channels: Optional[int] = None,
    reduction: int = 4
):
    super().__init__()
    
    freq_channels = freq_channels or channels
    
    # 空域特征处理 - 确保输入输出通道匹配
    self.spatial_proj = nn.Sequential(
        nn.Conv2d(channels, channels, 1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True)
    )
    
    # 频域特征处理 - 添加通道投影
    self.freq_proj = nn.Sequential(
        nn.Conv2d(freq_channels, channels, 1, bias=False),  # 投影到 channels
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True)
    )
    
    # ... 其余代码不变
```

---

### 问题 3: SpatialBranch 输出维度

**修复方案**:

```python
# models/recognition/spatial_branch.py
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    返回多尺度特征，确保通道数正确
    """
    # Stem
    x = self.stem(x)
    
    # Stages
    feat1 = self.layer1(x)   # (B, 64, 28, 28)
    feat2 = self.layer2(feat1)  # (B, 128, 14, 14)
    feat3 = self.layer3(feat2)  # (B, 256, 7, 7)
    feat4 = self.layer4(feat3)  # (B, 512, 3, 3)
    
    return (feat1, feat2, feat3, feat4)
```

---

## 🔍 调试步骤

### 步骤 1: 打印维度信息

```python
# 在关键位置添加调试代码
def forward(self, x):
    print(f"Input shape: {x.shape}")
    
    # Backbone
    features = self.backbone(x)
    for i, feat in enumerate(features):
        print(f"Backbone P{i+3} shape: {feat.shape}")
    
    # Neck
    fused = self.neck(features)
    for i, feat in enumerate(fused):
        print(f"Neck F{i+2} shape: {feat.shape}")
    
    # Head
    outputs = self.head(fused)
    for key, preds in outputs.items():
        for i, pred in enumerate(preds):
            print(f"Head {key}[{i}] shape: {pred.shape}")
    
    return outputs
```

### 步骤 2: 单元测试

```python
def test_detector_forward():
    from models.detection import DKGA_Det
    
    model = DKGA_Det()
    model.eval()
    
    x = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Output: {outputs}")
    assert len(outputs) == 1  # batch_size
    print("✅ Detector forward test passed")
```

---

## 📝 修复清单

- [x] 修复 `detection/head.py` ConvBNAct 导入
- [ ] 修复 `detection/neck.py` P2 层通道数
- [ ] 修复 `recognition/fusion.py` 频域通道投影
- [ ] 修复 `recognition/spatial_branch.py` 输出维度
- [ ] 运行完整推理测试
- [ ] 验证修复效果

---

## 🎯 验证标准

```bash
# 运行测试
python tools/test_inference.py

# 期望输出:
# ============================================================
# DDFD-FaceRec 推理测试
# ============================================================
# 测试人脸检测器
#   检测人脸数：1
#   平均推理时间：4.2ms
# 测试人脸识别器
#   特征维度：(512,)
#   平均推理时间：3.8ms
# 测试特征匹配器
#   相似度：0.1527
#   自相似度：1.0000
# 测试端到端流水线
#   检测到 1 张人脸
#   特征提取：成功
# ============================================================
# ✅ 所有测试通过!
```

---

**最后更新**: 2026 年 3 月 7 日  
**状态**: 修复中
