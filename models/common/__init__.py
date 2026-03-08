"""
通用模块包
"""

from .utils import (
    set_seed,
    get_device,
    count_parameters,
    get_flops,
    freeze_layer,
    unfreeze_layer,
    freeze_bn,
    is_dist_available_and_initialized,
    get_rank,
    get_world_size,
    is_main_process,
    synchronize,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    ProgressMeter,
    normalize_image,
    denormalize_image,
    xyxy_to_xywh,
    xywh_to_xyxy,
    clip_bbox,
    bbox_area,
    bbox_iou,
    load_yaml_config,
    merge_configs,
    to_numpy,
    to_tensor,
    format_time,
    format_number,
)

from .dcnv2 import (
    DeformConv2d,
    DeformableKeypointConv,
    deformable_conv2d_native,
    make_divisible,
)

from .backbone_utils import ConvBNAct

from .attention import (
    SEBlock,
    CBAM,
    ChannelAttention,
    SpatialAttention,
    ECABlock,
    FrequencyGatedAttention,
    DCTTransform,
    SelfAttention,
    CoordinateAttention,
    build_attention,
)

__all__ = [
    # utils
    "set_seed",
    "get_device",
    "count_parameters",
    "get_flops",
    "freeze_layer",
    "unfreeze_layer",
    "freeze_bn",
    "is_dist_available_and_initialized",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "synchronize",
    "save_checkpoint",
    "load_checkpoint",
    "AverageMeter",
    "ProgressMeter",
    "normalize_image",
    "denormalize_image",
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "clip_bbox",
    "bbox_area",
    "bbox_iou",
    "load_yaml_config",
    "merge_configs",
    "to_numpy",
    "to_tensor",
    "format_time",
    "format_number",
    # dcnv2
    "DeformConv2d",
    "DeformableKeypointConv",
    "deformable_conv2d_native",
    "make_divisible",
    # backbone_utils
    "ConvBNAct",
    # attention
    "SEBlock",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "ECABlock",
    "FrequencyGatedAttention",
    "DCTTransform",
    "SelfAttention",
    "CoordinateAttention",
    "build_attention",
]
