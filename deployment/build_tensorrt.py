"""
TensorRT 模型构建工具

将 ONNX 模型转换为 TensorRT 引擎
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def build_tensorrt_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    max_workspace_size: int = 4294967296,  # 4GB
    min_batch_size: int = 1,
    opt_batch_size: int = 8,
    input_shape: Optional[Tuple[int, int, int]] = None,
):
    """
    构建 TensorRT 引擎
    
    Args:
        onnx_path: ONNX 模型路径
        output_path: TensorRT 引擎输出路径
        precision: 精度模式 (fp32/fp16/int8)
        max_batch_size: 最大 batch size
        max_workspace_size: 最大工作空间 (bytes)
        min_batch_size: 最小 batch size
        opt_batch_size: 最优 batch size
        input_shape: 输入形状 (C, H, W)
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT is required. Install from NVIDIA.")
    
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
            raise RuntimeError("Failed to parse ONNX model")
    
    logging.info(f"ONNX model parsed successfully. Layers: {network.num_layers}")
    
    # 配置 builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # 设置精度
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # 需要校准器
        logging.warning("INT8 mode requires calibration. Using default calibrator.")
    
    # 设置动态形状 (如果支持)
    profile = builder.create_optimization_profile()
    
    # 获取输入名称和形状
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_dtype = input_tensor.dtype
    
    if input_shape is None:
        input_shape = tuple(input_tensor.shape[1:])
    
    C, H, W = input_shape
    
    # 设置动态形状范围
    profile.set_shape(
        input_name,
        (min_batch_size, C, H, W),  # 最小
        (opt_batch_size, C, H, W),  # 最优
        (max_batch_size, C, H, W)   # 最大
    )
    config.add_optimization_profile(profile)
    
    # 构建引擎
    logging.info("Building engine...")
    
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # 保存引擎
    logging.info(f"Saving engine to: {output_path}")
    
    with open(output_path, 'wb') as f:
        f.write(engine)
    
    logging.info("TensorRT engine built successfully!")
    
    # 打印信息
    engine_size = os.path.getsize(output_path) / 1024 / 1024
    logging.info(f"Engine size: {engine_size:.2f} MB")


class Int8Calibrator:
    """
    INT8 校准器
    """
    
    def __init__(
        self,
        calibration_data_path: str,
        cache_file: str,
        batch_size: int = 32,
        input_shape: Tuple[int, int, int] = (3, 640, 640)
    ):
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT is required.")
        
        self.trt = trt
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file
        
        # 加载校准数据
        self.calibration_data = self._load_calibration_data(calibration_data_path)
        self.data_index = 0
        
        # 创建输入 buffer
        self.input_buffer = trt.CudaPinnedMemory(
            batch_size * np.prod(input_shape) * 4  # float32
        )
    
    def _load_calibration_data(self, path: str):
        """加载校准数据"""
        # 这里简化处理，实际应从路径加载图像
        logging.info(f"Loading calibration data from: {path}")
        return []
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_batch(self, names, p_str=None) -> bool:
        """获取下一批校准数据"""
        if self.data_index >= len(self.calibration_data):
            return False
        
        # 填充输入 buffer
        # 实际实现需要从校准数据中读取
        
        self.data_index += 1
        return True
    
    def read_calibration_cache(self) -> bytes:
        """读取校准缓存"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return b''
    
    def write_calibration_cache(self, cache: bytes) -> None:
        """写入校准缓存"""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="ONNX model path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="TensorRT engine output path"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Precision mode"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size"
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        default=4294967296,
        help="Maximum workspace size in bytes"
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Input shape (C,H,W), e.g., 3,640,640"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # 解析输入形状
    input_shape = None
    if args.input_shape:
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
    
    build_tensorrt_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        max_workspace_size=args.workspace_size,
        input_shape=input_shape
    )


if __name__ == "__main__":
    main()
