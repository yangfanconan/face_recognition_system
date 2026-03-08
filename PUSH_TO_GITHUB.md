# 📤 GitHub 推送指南

## 当前状态

✅ Git 仓库已初始化
✅ 代码已提交 (2 commits, 81 files)
✅ 远程仓库已配置

**远程仓库**: `https://github.com/yangfanconan/face_recognition_system.git`

---

## 推送方法

### 方法 1: HTTPS 推送 (推荐)

```bash
cd /Users/yangfan/face_recognition_system

# 确保使用 HTTPS
git remote set-url origin https://github.com/yangfanconan/face_recognition_system.git

# 推送 (会提示输入 GitHub 密码或 Token)
git push -u origin main
```

**注意**: GitHub 现在使用 Personal Access Token 而不是密码

**获取 Token**:
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成后复制 Token
5. 推送时使用 Token 作为密码

---

### 方法 2: SSH 推送

```bash
# 1. 生成 SSH 密钥 (如果没有)
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 查看公钥
cat ~/.ssh/id_ed25519.pub

# 3. 复制公钥到 GitHub
# 访问 https://github.com/settings/keys
# 点击 "New SSH key" 并粘贴

# 4. 推送
git remote set-url origin git@github.com:yangfanconan/face_recognition_system.git
git push -u origin main
```

---

### 方法 3: GitHub Desktop (图形界面)

1. 下载 GitHub Desktop: https://desktop.github.com
2. 登录 GitHub 账号
3. File → Add Local Repository
4. 选择 `/Users/yangfan/face_recognition_system`
5. 点击 Publish repository

---

### 方法 4: 命令行凭证存储

```bash
# 配置凭证存储
git config --global credential.helper store

# 然后推送 (会保存凭证)
git push -u origin main
```

---

## 验证推送

推送成功后，访问：
https://github.com/yangfanconan/face_recognition_system

应该能看到所有文件。

---

## 常见问题

### Q: "Connection timed out"
**原因**: 网络连接问题

**解决**:
- 检查网络连接
- 尝试使用代理
- 稍后重试

### Q: "Permission denied"
**原因**: 认证失败

**解决**:
- 使用 Personal Access Token
- 或配置 SSH 密钥

### Q: "Repository not found"
**原因**: 仓库不存在

**解决**:
- 先在 GitHub 创建空仓库
- 或确保用户名正确

---

## 仓库信息

**提交历史**:
```
70ba002 docs: Add GitHub setup guide
e23fc31 Initial commit: DDFD-FaceRec v1.0
```

**文件统计**:
- 81 个文件
- 20,000+ 行代码
- 8 个文档

---

## 后续操作

推送成功后：

1. 更新 README.md 添加 GitHub 链接
2. 添加 License 文件
3. 配置 GitHub Actions (CI/CD)
4. 添加 Release

```bash
# 后续更新
git add .
git commit -m "描述更改"
git push origin main
```

---

**创建时间**: 2026 年 3 月 7 日
