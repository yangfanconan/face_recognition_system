"""
识别模型包
"""

from .spatial_branch import (
    BasicBlock,
    Bottleneck,
    SpatialBranch,
    SpatialBranchTiny,
    build_spatial_branch,
)

from .frequency_branch import (
    DCTTransform2D,
    FrequencyConvBlock,
    FrequencyBranch,
    FrequencyEnhancementModule,
    LowLightEnhancementBranch,
    build_frequency_branch,
)

from .fusion import (
    FrequencyGatedFusion,
    MultiHeadAttention,
    Mlp,
    TransformerBlock,
    TransformerEncoder,
    FeaturePyramidFusion,
    build_fusion_module,
    build_transformer_encoder,
)

from .head import (
    IdentityDisentangledHead,
    ArcFaceHead,
    AttributeHead,
    RecognitionHead,
    build_recognition_head,
)

from .losses import (
    AdaArcLoss,
    OrthogonalLoss,
    AttributeDisentanglementLoss,
    CenterLoss,
    RecognitionLoss,
    build_recognition_loss,
)

from .dfdf_rec import (
    DDFD_Rec,
    DDFD_Rec_Tiny,
    build_recognizer,
)

__all__ = [
    # Spatial branch
    "BasicBlock",
    "Bottleneck",
    "SpatialBranch",
    "SpatialBranchTiny",
    "build_spatial_branch",
    # Frequency branch
    "DCTTransform2D",
    "FrequencyConvBlock",
    "FrequencyBranch",
    "FrequencyEnhancementModule",
    "LowLightEnhancementBranch",
    "build_frequency_branch",
    # Fusion
    "FrequencyGatedFusion",
    "MultiHeadAttention",
    "Mlp",
    "TransformerBlock",
    "TransformerEncoder",
    "FeaturePyramidFusion",
    "build_fusion_module",
    "build_transformer_encoder",
    # Head
    "IdentityDisentangledHead",
    "ArcFaceHead",
    "AttributeHead",
    "RecognitionHead",
    "build_recognition_head",
    # Losses
    "AdaArcLoss",
    "OrthogonalLoss",
    "AttributeDisentanglementLoss",
    "CenterLoss",
    "RecognitionLoss",
    "build_recognition_loss",
    # Main models
    "DDFD_Rec",
    "DDFD_Rec_Tiny",
    "build_recognizer",
]
