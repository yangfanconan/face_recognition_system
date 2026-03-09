#!/usr/bin/env python3
"""
基于 Flask 的简单 Web 演示
"""

import os
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template_string, request, jsonify
import base64
import numpy as np
import cv2

# 导入检测模型
from inference.detector import Detector
from inference.recognizer import Recognizer
from inference.matcher import Matcher

# 初始化模型
print("Loading models...")
detector = Detector(score_thresh=0.5)
recognizer = Recognizer()
matcher = Matcher(threshold=0.6)
print("Models loaded successfully!")

app = Flask(__name__)

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DDFD-FaceRec Demo</title>
    <style>
        body { font-family: Arial; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #667eea; text-align: center; }
        .tabs { display: flex; gap: 10px; margin: 20px 0; justify-content: center; }
        .tab { padding: 12px 24px; background: #667eea; color: white; border: none; cursor: pointer; border-radius: 5px; font-size: 16px; }
        .tab:hover { background: #764ba2; }
        .tab.active { background: #764ba2; }
        .tab-content { display: none; padding: 30px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .tab-content.active { display: block; }
        .upload-area { border: 3px dashed #667eea; padding: 50px; text-align: center; cursor: pointer; border-radius: 10px; margin: 20px 0; }
        .upload-area:hover { background: #f5f3ff; }
        img { max-width: 100%; margin-top: 10px; border-radius: 5px; }
        .result { margin-top: 20px; padding: 20px; background: #f9fafb; border-radius: 5px; }
        .btn { background: #667eea; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; margin-top: 15px; font-size: 16px; }
        .btn:hover { background: #764ba2; }
        .info { background: #1f2937; color: #4ade80; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; margin-top: 15px; }
        .two-images { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .two-images { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <h1>🎯 DDFD-FaceRec 人脸识别演示系统</h1>
    <p style="text-align: center; color: #666;">基于训练好的 DKGA-Det 检测模型和 DDFD-Rec 识别模型</p>
    
    <div class="tabs">
        <button class="tab active" onclick="showTab('detection')">📷 人脸检测</button>
        <button class="tab" onclick="showTab('extract')">🔍 特征提取</button>
        <button class="tab" onclick="showTab('verify')">👥 1:1 验证</button>
        <button class="tab" onclick="showTab('about')">ℹ️ 关于</button>
    </div>
    
    <div id="detection" class="tab-content active">
        <h2>📷 人脸检测</h2>
        <p>上传图像，自动检测其中的人脸位置</p>
        <div class="upload-area" onclick="document.getElementById('file1').click()">
            <p>📁 点击上传图像</p>
            <input type="file" id="file1" accept="image/*" style="display:none;" onchange="previewImage(this, 'preview1')">
        </div>
        <img id="preview1" style="display:none;">
        <button class="btn" onclick="detect()">开始检测</button>
        <div id="result1" class="result" style="display:none;"></div>
    </div>
    
    <div id="extract" class="tab-content">
        <h2>🔍 特征提取</h2>
        <p>提取人脸的 512 维特征向量</p>
        <div class="upload-area" onclick="document.getElementById('file2').click()">
            <p>📁 点击上传图像</p>
            <input type="file" id="file2" accept="image/*" style="display:none;" onchange="previewImage(this, 'preview2')">
        </div>
        <img id="preview2" style="display:none;">
        <button class="btn" onclick="extract()">提取特征</button>
        <div id="result2" class="result" style="display:none;"></div>
    </div>
    
    <div id="verify" class="tab-content">
        <h2>👥 1:1 人脸验证</h2>
        <p>验证两张图像是否为同一人</p>
        <div class="two-images">
            <div>
                <div class="upload-area" onclick="document.getElementById('file3a').click()">
                    <p>📁 上传图像 1</p>
                    <input type="file" id="file3a" accept="image/*" style="display:none;" onchange="previewImage(this, 'preview3a')">
                </div>
                <img id="preview3a" style="display:none;">
            </div>
            <div>
                <div class="upload-area" onclick="document.getElementById('file3b').click()">
                    <p>📁 上传图像 2</p>
                    <input type="file" id="file3b" accept="image/*" style="display:none;" onchange="previewImage(this, 'preview3b')">
                </div>
                <img id="preview3b" style="display:none;">
            </div>
        </div>
        <button class="btn" onclick="verify()" style="width: 100%;">开始验证</button>
        <div id="result3" class="result" style="display:none;"></div>
    </div>
    
    <div id="about" class="tab-content">
        <h2>ℹ️ 关于</h2>
        <p><strong>DDFD-FaceRec</strong> - 双分支特征解耦人脸识别系统</p>
        <p><strong>版本：</strong>v1.0-beta</p>
        <p><strong>检测模型：</strong>DKGA-Det (64.82M 参数)</p>
        <p><strong>识别模型：</strong>DDFD-Rec (27.99M 参数)</p>
        <p><strong>特征维度：</strong>512-d</p>
        <p><strong>GitHub：</strong><a href="https://github.com/yangfanconan/face_recognition_system" target="_blank">yangfanconan/face_recognition_system</a></p>
    </div>
    
    <script>
        function showTab(id) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            event.target.classList.add('active');
        }
        
        function previewImage(input, previewId) {
            const file = input.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.getElementById(previewId);
                    img.src = e.target.result;
                    img.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        }
        
        function detect() {
            const file = document.getElementById('file1').files[0];
            if (!file) { alert('请先上传图像'); return; }
            
            const formData = new FormData();
            formData.append('image', file);
            
            document.getElementById('result1').innerHTML = '正在检测...';
            document.getElementById('result1').style.display = 'block';
            
            fetch('/api/detect', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                const result = document.getElementById('result1');
                result.innerHTML = '<h3>检测结果</h3>' +
                    '<img src="data:image/jpeg;base64,' + data.image + '" />' +
                    '<div class="info">' + data.info + '</div>';
            })
            .catch(err => { document.getElementById('result1').innerHTML = '错误：' + err; });
        }
        
        function extract() {
            const file = document.getElementById('file2').files[0];
            if (!file) { alert('请先上传图像'); return; }
            
            const formData = new FormData();
            formData.append('image', file);
            
            document.getElementById('result2').innerHTML = '正在提取...';
            document.getElementById('result2').style.display = 'block';
            
            fetch('/api/extract', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                const result = document.getElementById('result2');
                result.innerHTML = '<h3>特征信息</h3><div class="info">' + data.info + '</div>';
            })
            .catch(err => { document.getElementById('result2').innerHTML = '错误：' + err; });
        }
        
        function verify() {
            const file1 = document.getElementById('file3a').files[0];
            const file2 = document.getElementById('file3b').files[0];
            if (!file1 || !file2) { alert('请上传两张图像'); return; }
            
            const formData = new FormData();
            formData.append('image1', file1);
            formData.append('image2', file2);
            
            document.getElementById('result3').innerHTML = '正在验证...';
            document.getElementById('result3').style.display = 'block';
            
            fetch('/api/verify', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                const result = document.getElementById('result3');
                const color = data.is_same ? '#10b981' : '#ef4444';
                result.innerHTML = '<h3>验证结果</h3>' +
                    '<div style="font-size: 24px; color: ' + color + '; font-weight: bold;">' + 
                    (data.is_same ? '✅ 同一人' : '❌ 不同人') + '</div>' +
                    '<div class="info">' + data.info + '</div>';
            })
            .catch(err => { document.getElementById('result3').innerHTML = '错误：' + err; });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect', methods=['POST'])
def detect():
    """人脸检测 API"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # 调用真实检测模型
    detections = detector.detect(img)
    
    # 绘制检测结果
    result_img = img.copy()
    for i, det in enumerate(detections):
        bbox = det['bbox']
        score = det['score']
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_img, f'{score:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    info = f"检测到 {len(detections)} 张人脸\n"
    for i, det in enumerate(detections):
        info += f"人脸{i+1}: 置信度={det['score']:.4f}, 位置=[{det['bbox']}]\n"
    
    return jsonify({
        'image': img_base64,
        'info': info,
        'count': len(detections)
    })

@app.route('/api/extract', methods=['POST'])
def extract():
    """特征提取 API"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # 检测人脸
    detections = detector.detect(img)
    
    if len(detections) == 0:
        return jsonify({'error': 'No face detected'}), 404
    
    # 提取第一个检测到的人脸特征
    bbox = detections[0]['bbox']
    feature = recognizer.extract(img, bbox)
    
    info = f"特征维度：{feature.shape}\n"
    info += f"特征范数：{np.linalg.norm(feature):.4f}\n"
    info += f"特征 (前 20 维): {feature[:20]}\n"
    
    return jsonify({
        'info': info,
        'feature_shape': feature.shape,
        'face_count': len(detections)
    })

@app.route('/api/verify', methods=['POST'])
def verify():
    """1:1 人脸验证 API"""
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Need two images'}), 400
    
    file1 = request.files['image1']
    file2 = request.files['image2']
    
    img1 = np.frombuffer(file1.read(), np.uint8)
    img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
    
    img2 = np.frombuffer(file2.read(), np.uint8)
    img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)
    
    if img1 is None or img2 is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # 检测人脸
    dets1 = detector.detect(img1)
    dets2 = detector.detect(img2)
    
    if len(dets1) == 0 or len(dets2) == 0:
        return jsonify({'error': 'No face detected in one or both images'}), 404
    
    # 提取特征
    bbox1 = dets1[0]['bbox']
    bbox2 = dets2[0]['bbox']
    
    feat1 = recognizer.extract(img1, bbox1)
    feat2 = recognizer.extract(img2, bbox2)
    
    # 比对
    is_same, similarity = matcher.verify(feat1, feat2)
    
    result = "同一人" if is_same else "不同人"
    info = f"相似度：{similarity:.4f}\n"
    info += f"阈值：{matcher.threshold}\n"
    info += f"结果：{result}\n"
    
    return jsonify({
        'info': info,
        'similarity': float(similarity),
        'is_same': bool(is_same)
    })

if __name__ == "__main__":
    print("Starting Flask server...")
    print("访问地址：http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
