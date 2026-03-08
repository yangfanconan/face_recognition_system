"""
推理模块包
"""

from .detector import Detector
from .recognizer import Recognizer, FaceAligner
from .matcher import (
    Matcher,
    FaceVerifier,
    FaceSearcher,
    QualityAssessor,
    cosine_similarity,
    weighted_cosine_similarity,
)
from .pipeline import FaceRecognitionPipeline, BatchProcessor, build_pipeline

from .index.hnsw_index import HNSWIndex, FaissIndex, build_index

__all__ = [
    # Detector
    "Detector",
    # Recognizer
    "Recognizer",
    "FaceAligner",
    # Matcher
    "Matcher",
    "FaceVerifier",
    "FaceSearcher",
    "QualityAssessor",
    "cosine_similarity",
    "weighted_cosine_similarity",
    # Pipeline
    "FaceRecognitionPipeline",
    "BatchProcessor",
    "build_pipeline",
    # Index
    "HNSWIndex",
    "FaissIndex",
    "build_index",
]
