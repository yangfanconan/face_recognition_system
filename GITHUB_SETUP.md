# 📦 推送到 GitHub 指南

## 1. 在 GitHub 上创建新仓库

访问 https://github.com/new 创建新仓库

**仓库名称**: `ddfd-face-rec` 或 `face-recognition-system`

**可见性**: 公开 (Public) 或 私有 (Private)

**不要** 初始化 README、.gitignore 或 license (我们已经有了)

---

## 2. 添加远程仓库并推送

```bash
cd /Users/yangfan/face_recognition_system

# 添加远程仓库 (替换为你的 GitHub 用户名和仓库名)
git remote add origin https://github.com/YOUR_USERNAME/ddfd-face-rec.git

# 或者使用 SSH (如果你配置了 SSH 密钥)
git remote add origin git@github.com:YOUR_USERNAME/ddfd-face-rec.git

# 查看远程仓库
git remote -v

# 推送到 GitHub
git branch -M main
git push -u origin main
```

---

## 3. 验证推送

访问你的 GitHub 仓库页面，确认文件已上传：
```
https://github.com/YOUR_USERNAME/ddfd-face-rec
```

---

## 4. 后续更新

```bash
# 提交更改
git add .
git commit -m "描述你的更改"

# 推送到 GitHub
git push origin main
```

---

## 5. 常见问题

### Q: 提示权限错误？
```bash
# 使用 HTTPS
git remote set-url origin https://github.com/YOUR_USERNAME/ddfd-face-rec.git

# 或使用 SSH
git remote set-url origin git@github.com:YOUR_USERNAME/ddfd-face-rec.git
```

### Q: 仓库已存在？
```bash
# 如果本地已有远程仓库
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/ddfd-face-rec.git
```

### Q: 推送大文件失败？
```bash
# 安装 Git LFS
brew install git-lfs  # Mac
# 或
sudo apt install git-lfs  # Linux

# 初始化
git lfs install

# 跟踪大文件
git lfs track "*.pth"
git lfs track "*.onnx"
git lfs track "*.trt"

# 重新添加并提交
git add .gitattributes
git commit -m "Configure Git LFS"
```

---

## 6. 仓库信息

**提交统计**:
- 提交数：1
- 文件数：80
- 代码行数：20,000+

**主要文件**:
- `README.md` - 项目说明
- `PROJECT_DESIGN.md` - 技术方案
- `QUICKSTART.md` - 快速开始
- `models/` - 模型代码
- `inference/` - 推理服务
- `tools/` - 训练和评估脚本
- `configs/` - 配置文件

---

**创建时间**: 2026 年 3 月 7 日
**版本**: v1.0
