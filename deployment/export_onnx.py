"""
ONNX 模型导出工具

将 PyTorch 模型导出为 ONNX 格式
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.onnx


def export_detector(
    checkpoint_path: str,
    output_path: str,
    model_type: str = "dkga_det",
    input_size: int = 640,
    opset_version: int = 13,
    dynamic_axes: bool = True
):
    """
    导出检测模型为 ONNX
    
    Args:
        checkpoint_path: PyTorch 权重路径
        output_path: ONNX 输出路径
        model_type: 模型类型
        input_size: 输入尺寸
        opset_version: ONNX opset 版本
        dynamic_axes: 是否支持动态轴
    """
    from models.detection import build_detector
    
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    
    # 加载模型
    model = build_detector(model_name=model_type)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # 设置动态轴
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        }
        for i, name in enumerate(['cls_output', 'reg_output', 'kpt_output']):
            dynamic_axes_dict[name] = {0: 'batch_size'}
    else:
        dynamic_axes_dict = None
    
    # 导出
    logging.info(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['cls_output', 'reg_output', 'kpt_output'],
        dynamic_axes=dynamic_axes_dict,
        verbose=False
    )
    
    # 验证导出
    logging.info("Verifying ONNX model...")
    
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # 优化模型
    logging.info("Optimizing ONNX model...")
    
    try:
        from onnxsim import simplify
        model_simp, check = simplify(output_path)
        
        if check:
            sim_path = output_path.replace('.onnx', '_sim.onnx')
            onnx.save(model_simp, sim_path)
            logging.info(f"Saved simplified model to: {sim_path}")
    except ImportError:
        logging.warning("onnxsim not installed. Install with: pip install onnxsim")
    
    logging.info(f"Export completed: {output_path}")
    
    # 打印模型信息
    model_size = os.path.getsize(output_path) / 1024 / 1024
    logging.info(f"Model size: {model_size:.2f} MB")


def export_recognizer(
    checkpoint_path: str,
    output_path: str,
    model_type: str = "ddfd_rec",
    input_size: int = 112,
    opset_version: int = 13,
    dynamic_axes: bool = True
):
    """
    导出识别模型为 ONNX
    
    Args:
        checkpoint_path: PyTorch 权重路径
        output_path: ONNX 输出路径
        model_type: 模型类型
        input_size: 输入尺寸
        opset_version: ONNX opset 版本
    """
    from models.recognition import build_recognizer
    
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    
    # 加载模型
    model = build_recognizer(model_type=model_type)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # 导出
    logging.info(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['feature'],
        dynamic_axes={'input': {0: 'batch_size'}, 'feature': {0: 'batch_size'}} if dynamic_axes else None,
        verbose=False
    )
    
    # 验证
    logging.info("Verifying ONNX model...")
    
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    logging.info(f"Export completed: {output_path}")
    
    model_size = os.path.getsize(output_path) / 1024 / 1024
    logging.info(f"Model size: {model_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["detection", "recognition"],
        required=True,
        help="Model type"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="PyTorch checkpoint path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="ONNX output path"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Input size (640 for detection, 112 for recognition)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.model == "detection":
        export_detector(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_size=args.input_size or 640,
            opset_version=args.opset
        )
    else:
        export_recognizer(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_size=args.input_size or 112,
            opset_version=args.opset
        )


if __name__ == "__main__":
    main()
