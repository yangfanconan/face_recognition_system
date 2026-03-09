#!/usr/bin/env python3
"""
简化版 Web 演示 - 用于快速测试
"""

import os
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

def detect_face(image):
    """人脸检测"""
    if image is None:
        return None, "请上传图像"
    
    # 这里应该调用检测模型
    # 简化版本：返回占位信息
    return image, f"检测到人脸 (演示模式)"

def verify_faces(img1, img2):
    """1:1 验证"""
    if img1 is None or img2 is None:
        return None, "请上传两张图像"
    
    return img1, f"验证结果：待实现 (演示模式)"

def search_face(image):
    """1:N 搜索"""
    if image is None:
        return None, "请上传图像"
    
    return image, f"搜索结果：待实现 (演示模式)"

# 创建界面
with gr.Blocks(title="DDFD-FaceRec Demo") as demo:
    gr.Markdown("# 🎯 DDFD-FaceRec 人脸识别演示")
    
    with gr.Tabs():
        # 人脸检测
        with gr.TabItem("📷 人脸检测"):
            with gr.Row():
                detect_input = gr.Image(label="输入图像", type="pil")
                detect_output = gr.Image(label="检测结果")
                detect_info = gr.Textbox(label="信息")
            
            detect_btn = gr.Button("开始检测", variant="primary")
            detect_btn.click(
                fn=detect_face,
                inputs=[detect_input],
                outputs=[detect_output, detect_info]
            )
        
        # 1:1 验证
        with gr.TabItem("👥 1:1 验证"):
            with gr.Row():
                v1 = gr.Image(label="图像 1")
                v2 = gr.Image(label="图像 2")
            v_result = gr.Image(label="结果")
            v_info = gr.Textbox(label="信息")
            v_btn = gr.Button("验证", variant="primary")
            v_btn.click(
                fn=verify_faces,
                inputs=[v1, v2],
                outputs=[v_result, v_info]
            )
        
        # 1:N 搜索
        with gr.TabItem("🔎 1:N 搜索"):
            s_input = gr.Image(label="查询图像")
            s_result = gr.Image(label="结果")
            s_info = gr.Textbox(label="信息")
            s_btn = gr.Button("搜索", variant="primary")
            s_btn.click(
                fn=search_face,
                inputs=[s_input],
                outputs=[s_result, s_info]
            )
    
    gr.Markdown("---")
    gr.Markdown("**GitHub**: https://github.com/yangfanconan/face_recognition_system")

if __name__ == "__main__":
    print("Starting Gradio server...")
    print("访问地址：http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
