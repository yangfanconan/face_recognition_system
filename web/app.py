#!/usr/bin/env python3
"""
人脸识别 Web 演示应用

提供完整的 Web 界面用于测试：
- 人脸检测
- 特征提取
- 1:1 人脸验证
- 1:N 人脸搜索
"""

import os
import sys
import io
import base64
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
from PIL import Image
import gradio as gr

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.detector import Detector
from inference.recognizer import Recognizer
from inference.matcher import Matcher
from inference.index.hnsw_index import HNSWIndex


# ============================================
# 全局变量
# ============================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 初始化模型
print("Loading models...")
detector = Detector(score_thresh=0.5, device=device)
recognizer = Recognizer(device=device)
matcher = Matcher(threshold=0.6, device=device)

# 人脸库 (用于 1:N 搜索)
face_database = []
face_features = []
hnsw_index = None

print("Models loaded successfully!")


# ============================================
# 工具函数
# ============================================

def load_image(image):
    """加载图像"""
    if image is None:
        return None
    
    # Gradio 传递的是 PIL Image
    if isinstance(image, Image.Image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image
    
    return img


def draw_faces(image, detections):
    """在图像上绘制人脸框"""
    result = image.copy()
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        score = det['score']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # 绘制边框
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"Face {i+1}: {score:.2f}"
        cv2.putText(result, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result


def image_to_base64(image):
    """图像转 base64"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


# ============================================
# 核心功能
# ============================================

def detect_faces(image):
    """人脸检测"""
    img = load_image(image)
    if img is None:
        return None, "No image provided", []
    
    # 检测
    detections = detector.detect(img)
    
    # 绘制结果
    result = draw_faces(img, detections)
    
    # 信息
    info = f"Detected {len(detections)} face(s)\n"
    for i, det in enumerate(detections):
        bbox = det['bbox']
        score = det['score']
        info += f"Face {i+1}: bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], score={score:.4f}\n"
    
    return result, info, detections


def extract_feature(image, face_index=0):
    """特征提取"""
    img = load_image(image)
    if img is None:
        return None, "No image provided", None
    
    # 先检测
    detections = detector.detect(img)
    
    if len(detections) == 0:
        return img, "No face detected", None
    
    if face_index >= len(detections):
        return img, f"Face index {face_index} out of range (0-{len(detections)-1})", None
    
    # 提取特征
    bbox = detections[face_index]['bbox']
    feature = recognizer.extract(img, bbox)
    
    # 绘制结果
    result = draw_faces(img, [detections[face_index]])
    
    # 信息
    info = f"Face {face_index+1} detected\n"
    info += f"Feature shape: {feature.shape}\n"
    info += f"Feature norm: {np.linalg.norm(feature):.4f}\n"
    info += f"Feature (first 10): {feature[:10]}\n"
    
    return result, info, feature


def verify_faces(image1, image2, face_idx1=0, face_idx2=0):
    """1:1 人脸验证"""
    img1 = load_image(image1)
    img2 = load_image(image2)
    
    if img1 is None or img2 is None:
        return None, "Please provide both images", None
    
    # 检测人脸
    dets1 = detector.detect(img1)
    dets2 = detector.detect(img2)
    
    if len(dets1) == 0 or len(dets2) == 0:
        return img1, "No face detected in one or both images", None
    
    if face_idx1 >= len(dets1) or face_idx2 >= len(dets2):
        return img1, "Face index out of range", None
    
    # 提取特征
    bbox1 = dets1[face_idx1]['bbox']
    bbox2 = dets2[face_idx2]['bbox']
    
    feat1 = recognizer.extract(img1, bbox1)
    feat2 = recognizer.extract(img2, bbox2)
    
    # 比对
    is_same, similarity = matcher.verify(feat1, feat2)
    
    # 绘制结果
    result1 = draw_faces(img1, [dets1[face_idx1]])
    result2 = draw_faces(img2, [dets2[face_idx2]])
    
    # 合并显示
    h = min(result1.shape[0], result2.shape[0])
    result1 = cv2.resize(result1, (300, h))
    result2 = cv2.resize(result2, (300, h))
    result = np.hstack([result1, result2])
    
    # 添加相似度标签
    status = "SAME PERSON" if is_same else "DIFFERENT PERSON"
    color = (0, 255, 0) if is_same else (0, 0, 255)
    cv2.putText(result, f"{status}: {similarity:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # 信息
    info = f"Similarity: {similarity:.4f}\n"
    info += f"Threshold: {matcher.threshold}\n"
    info += f"Result: {status}\n"
    info += f"Face 1 norm: {np.linalg.norm(feat1):.4f}\n"
    info += f"Face 2 norm: {np.linalg.norm(feat2):.4f}\n"
    
    return result, info, similarity


def register_face(image, name, face_index=0):
    """注册人脸到图库"""
    global face_database, face_features, hnsw_index
    
    img = load_image(image)
    if img is None:
        return "No image provided"
    
    # 检测
    detections = detector.detect(img)
    
    if len(detections) == 0:
        return "No face detected"
    
    if face_index >= len(detections):
        return f"Face index {face_index} out of range"
    
    # 提取特征
    bbox = detections[face_index]['bbox']
    feature = recognizer.extract(img, bbox)
    
    # 添加到数据库
    face_database.append({
        'name': name,
        'image': image_to_base64(img),
        'bbox': bbox.tolist(),
        'timestamp': datetime.now().isoformat()
    })
    face_features.append(feature)
    
    # 重建索引
    if len(face_features) > 0:
        hnsw_index = HNSWIndex(dim=len(feature), max_elements=10000)
        features_array = np.vstack(face_features)
        ids = np.arange(len(face_features))
        hnsw_index.add(features_array, ids)
        matcher.init_search_index(hnsw_index)
    
    return f"Registered: {name} (Total: {len(face_database)} faces)"


def search_face(image, face_index=0, top_k=5):
    """1:N 人脸搜索"""
    global face_database, face_features, hnsw_index
    
    img = load_image(image)
    if img is None:
        return None, "No image provided", []
    
    if len(face_database) == 0:
        return img, "Face database is empty. Please register faces first.", []
    
    # 检测
    detections = detector.detect(img)
    
    if len(detections) == 0:
        return img, "No face detected", []
    
    if face_index >= len(detections):
        return img, f"Face index {face_index} out of range", []
    
    # 提取特征
    bbox = detections[face_index]['bbox']
    query_feature = recognizer.extract(img, bbox)
    
    # 搜索
    results = matcher.search(query_feature, top_k=top_k, threshold=0.4)
    
    # 绘制查询人脸
    result = draw_faces(img, [detections[face_index]])
    
    # 信息
    info = f"Query: Face {face_index+1}\n"
    info += f"Database size: {len(face_database)}\n"
    info += f"Top {top_k} results:\n\n"
    
    for i, res in enumerate(results):
        idx = res['id']
        sim = res['similarity']
        name = face_database[idx]['name']
        info += f"{i+1}. {name}: {sim:.4f}\n"
    
    return result, info, results


def clear_database():
    """清空人脸库"""
    global face_database, face_features, hnsw_index
    face_database = []
    face_features = []
    hnsw_index = None
    return "Face database cleared"


def get_database_info():
    """获取人脸库信息"""
    global face_database
    if len(face_database) == 0:
        return "Database is empty"
    
    info = f"Total faces: {len(face_database)}\n\n"
    for i, entry in enumerate(face_database):
        info += f"{i+1}. {entry['name']} ({entry['timestamp']})\n"
    
    return info


# ============================================
# Gradio 界面
# ============================================

def create_demo():
    """创建 Gradio 演示界面"""
    
    with gr.Blocks(title="DDFD-FaceRec Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 DDFD-FaceRec 人脸识别演示系统
        
        **Dual-Domain Feature Decoupling Face Recognition System**
        
        本演示提供完整的人脸识别功能测试：
        - ✅ 人脸检测
        - ✅ 特征提取
        - ✅ 1:1 人脸验证
        - ✅ 1:N 人脸搜索
        
        ---
        """)
        
        with gr.Tabs():
            # ========== 人脸检测 ==========
            with gr.TabItem("📷 人脸检测"):
                gr.Markdown("### 上传图像进行人脸检测")
                
                with gr.Row():
                    with gr.Column():
                        detect_input = gr.Image(label="输入图像", type="pil")
                        detect_btn = gr.Button("开始检测", variant="primary")
                    
                    with gr.Column():
                        detect_output = gr.Image(label="检测结果")
                        detect_info = gr.Textbox(label="检测信息", lines=8)
                
                detect_btn.click(
                    fn=detect_faces,
                    inputs=[detect_input],
                    outputs=[detect_output, detect_info]
                )
            
            # ========== 特征提取 ==========
            with gr.TabItem("🔍 特征提取"):
                gr.Markdown("### 提取人脸特征向量")
                
                with gr.Row():
                    with gr.Column():
                        extract_input = gr.Image(label="输入图像", type="pil")
                        extract_idx = gr.Number(label="人脸索引", value=0, precision=0)
                        extract_btn = gr.Button("提取特征", variant="primary")
                    
                    with gr.Column():
                        extract_output = gr.Image(label="检测结果")
                        extract_info = gr.Textbox(label="特征信息", lines=8)
                
                extract_btn.click(
                    fn=extract_feature,
                    inputs=[extract_input, extract_idx],
                    outputs=[extract_output, extract_info]
                )
            
            # ========== 1:1 验证 ==========
            with gr.TabItem("👥 1:1 人脸验证"):
                gr.Markdown("### 验证两张图像是否为同一人")
                
                with gr.Row():
                    verify_input1 = gr.Image(label="图像 1", type="pil")
                    verify_input2 = gr.Image(label="图像 2", type="pil")
                
                with gr.Row():
                    verify_idx1 = gr.Number(label="图像 1 人脸索引", value=0, precision=0)
                    verify_idx2 = gr.Number(label="图像 2 人脸索引", value=0, precision=0)
                    verify_btn = gr.Button("开始验证", variant="primary")
                
                with gr.Row():
                    verify_output = gr.Image(label="验证结果")
                    verify_info = gr.Textbox(label="验证信息", lines=8)
                
                verify_btn.click(
                    fn=verify_faces,
                    inputs=[verify_input1, verify_input2, verify_idx1, verify_idx2],
                    outputs=[verify_output, verify_info]
                )
            
            # ========== 1:N 搜索 ==========
            with gr.TabItem("🔎 1:N 人脸搜索"):
                gr.Markdown("""
                ### 在人脸库中搜索相似人脸
                
                **使用步骤**:
                1. 先注册人脸到数据库
                2. 上传查询图像进行搜索
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 注册人脸")
                        reg_input = gr.Image(label="注册图像", type="pil")
                        reg_name = gr.Textbox(label="姓名/ID", placeholder="输入姓名或 ID")
                        reg_idx = gr.Number(label="人脸索引", value=0, precision=0)
                        reg_btn = gr.Button("注册人脸", variant="primary")
                        reg_info = gr.Textbox(label="注册状态", lines=2)
                    
                    with gr.Column():
                        gr.Markdown("#### 搜索人脸")
                        search_input = gr.Image(label="查询图像", type="pil")
                        search_idx = gr.Number(label="人脸索引", value=0, precision=0)
                        search_topk = gr.Number(label="返回数量", value=5, precision=0)
                        search_btn = gr.Button("开始搜索", variant="primary")
                
                with gr.Row():
                    search_output = gr.Image(label="查询结果")
                    search_info = gr.Textbox(label="搜索信息", lines=10)
                
                reg_btn.click(
                    fn=register_face,
                    inputs=[reg_input, reg_name, reg_idx],
                    outputs=[reg_info]
                )
                
                search_btn.click(
                    fn=search_face,
                    inputs=[search_input, search_idx, search_topk],
                    outputs=[search_output, search_info]
                )
            
            # ========== 人脸库管理 ==========
            with gr.TabItem("📋 人脸库管理"):
                gr.Markdown("### 查看和管理已注册的人脸")
                
                with gr.Row():
                    db_info_btn = gr.Button("刷新数据库信息", variant="secondary")
                    db_clear_btn = gr.Button("清空数据库", variant="stop")
                
                db_info = gr.Textbox(label="数据库信息", lines=15)
                
                db_info_btn.click(fn=get_database_info, outputs=[db_info])
                db_clear_btn.click(fn=clear_database, outputs=[db_info])
        
        gr.Markdown("""
        ---
        **技术栈**: DDFD-Rec | DKGA-Det | HNSW | Gradio
        
        **模型版本**: v1.0-beta
        
        **GitHub**: https://github.com/yangfanconan/face_recognition_system
        """)
    
    return demo


# ============================================
# 主函数
# ============================================

def main():
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
