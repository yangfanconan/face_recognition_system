#!/usr/bin/env python3
"""
模型转换工具

支持:
- PyTorch -> ONNX
- ONNX -> TensorRT
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================
# PyTorch -> ONNX
# ============================================

def export_detector_onnx(
    checkpoint_path: str,
    output_path: str,
    input_size: int = 640,
    opset_version: int = 13,
    device: str = "cpu"
):
    """导出检测模型为 ONNX"""
    import torch
    from models.detection import build_detector
    
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    
    # 加载模型
    model = build_detector(model_name="dkga_det")
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
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
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'cls_output': {0: 'batch_size'},
            'reg_output': {0: 'batch_size'},
            'kpt_output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    # 验证
    logging.info("Verifying ONNX model...")
    
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    model_size = os.path.getsize(output_path) / 1024 / 1024
    logging.info(f"Export completed: {output_path} ({model_size:.2f} MB)")
    
    return True


def export_recognizer_onnx(
    checkpoint_path: str,
    output_path: str,
    input_size: int = 112,
    opset_version: int = 13,
    device: str = "cpu"
):
    """导出识别模型为 ONNX"""
    import torch
    from models.recognition import build_recognizer
    
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    
    # 加载模型
    model = build_recognizer(model_type="ddfd_rec")
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
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
        output_names=['feature'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'feature': {0: 'batch_size'}
        },
        verbose=False
    )
    
    # 验证
    logging.info("Verifying ONNX model...")
    
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    model_size = os.path.getsize(output_path) / 1024 / 1024
    logging.info(f"Export completed: {output_path} ({model_size:.2f} MB)")
    
    return True


# ============================================
# ONNX -> TensorRT
# ============================================

def build_tensorrt_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    max_workspace_size: int = 4294967296
):
    """构建 TensorRT 引擎"""
    try:
        import tensorrt as trt
    except ImportError:
        logging.error("TensorRT is required. Install from NVIDIA.")
        return False
    
    logging.info(f"Building TensorRT engine from: {onnx_path}")
    logging.info(f"Precision: {precision}")
    
    # 创建 logger 和 builder
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    
    # 解析 ONNX
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            logging.error("Failed to parse ONNX model:")
            for error in range(parser.num_errors):
                logging.error(parser.get_error(error))
            return False
    
    logging.info(f"ONNX model parsed. Layers: {network.num_layers}")
    
    # 配置 builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # 设置精度
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        logging.warning("INT8 mode requires calibration.")
    
    # 构建引擎
    logging.info("Building engine...")
    
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        logging.error("Failed to build TensorRT engine")
        return False
    
    # 保存引擎
    logging.info(f"Saving engine to: {output_path}")
    
    with open(output_path, 'wb') as f:
        f.write(engine)
    
    engine_size = os.path.getsize(output_path) / 1024 / 1024
    logging.info(f"TensorRT engine built: {output_path} ({engine_size:.2f} MB)")
    
    return True


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Model conversion tools")
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # ONNX 导出命令
    onnx_parser = subparsers.add_parser('onnx', help='Export to ONNX')
    onnx_parser.add_argument('--model', type=str, required=True,
                            choices=['detector', 'recognizer'],
                            help='Model type')
    onnx_parser.add_argument('--checkpoint', type=str, required=True,
                            help='PyTorch checkpoint path')
    onnx_parser.add_argument('--output', type=str, required=True,
                            help='ONNX output path')
    onnx_parser.add_argument('--input-size', type=int, default=None,
                            help='Input size (640 for detector, 112 for recognizer)')
    onnx_parser.add_argument('--opset', type=int, default=13,
                            help='ONNX opset version')
    onnx_parser.add_argument('--device', type=str, default='cpu',
                            help='Device (cpu/cuda)')
    
    # TensorRT 构建命令
    trt_parser = subparsers.add_parser('trt', help='Build TensorRT engine')
    trt_parser.add_argument('--onnx', type=str, required=True,
                           help='ONNX model path')
    trt_parser.add_argument('--output', type=str, required=True,
                           help='TensorRT engine output path')
    trt_parser.add_argument('--precision', type=str, default='fp16',
                           choices=['fp32', 'fp16', 'int8'],
                           help='Precision mode')
    trt_parser.add_argument('--max-batch', type=int, default=32,
                           help='Maximum batch size')
    trt_parser.add_argument('--workspace', type=int, default=4294967296,
                           help='Maximum workspace size in bytes')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.command == 'onnx':
        input_size = args.input_size or (640 if args.model == 'detector' else 112)
        
        if args.model == 'detector':
            success = export_detector_onnx(
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                input_size=input_size,
                opset_version=args.opset,
                device=args.device
            )
        else:
            success = export_recognizer_onnx(
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                input_size=input_size,
                opset_version=args.opset,
                device=args.device
            )
        
        sys.exit(0 if success else 1)
    
    elif args.command == 'trt':
        success = build_tensorrt_engine(
            onnx_path=args.onnx,
            output_path=args.output,
            precision=args.precision,
            max_batch_size=args.max_batch,
            max_workspace_size=args.workspace
        )
        
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
