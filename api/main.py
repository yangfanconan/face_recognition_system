"""
DDFD-FaceRec API 服务

FastAPI 人脸识别服务
"""

import os
import io
import time
import logging
from typing import Dict, List, Optional, Union
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import cv2

# 延迟导入推理模块
_pipeline = None


def get_pipeline():
    """懒加载推理流水线"""
    global _pipeline
    if _pipeline is None:
        from inference import FaceRecognitionPipeline
        _pipeline = FaceRecognitionPipeline()
    return _pipeline


# ============================================
# 数据模型
# ============================================

class DetectResponse(BaseModel):
    """检测响应"""
    success: bool
    message: str = ""
    data: Optional[Dict] = None
    inference_time_ms: float


class ExtractResponse(BaseModel):
    """特征提取响应"""
    success: bool
    message: str = ""
    data: Optional[Dict] = None
    inference_time_ms: float


class VerifyRequest(BaseModel):
    """验证请求"""
    feature1: List[float] = Field(..., description="特征向量 1")
    feature2: List[float] = Field(..., description="特征向量 2")
    threshold: float = Field(default=0.6, description="验证阈值")


class VerifyResponse(BaseModel):
    """验证响应"""
    success: bool
    is_same: bool
    similarity: float
    confidence: float
    inference_time_ms: float


class SearchRequest(BaseModel):
    """搜索请求"""
    feature: List[float] = Field(..., description="查询特征")
    top_k: int = Field(default=10, description="返回数量")
    threshold: float = Field(default=0.6, description="相似度阈值")
    gallery_id: Optional[str] = Field(default=None, description="图库 ID")


class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool
    results: List[Dict]
    count: int
    search_time_ms: float


class RegisterRequest(BaseModel):
    """注册请求"""
    person_id: str = Field(..., description="人员 ID")
    feature: List[float] = Field(..., description="特征向量")
    metadata: Optional[Dict] = Field(default=None, description="元数据")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    gpu_available: bool
    model_loaded: bool


# ============================================
# 应用生命周期
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logging.info("Starting DDFD-FaceRec API...")
    yield
    # 关闭时
    logging.info("Shutting down DDFD-FaceRec API...")


# ============================================
# 创建应用
# ============================================

app = FastAPI(
    title="DDFD-FaceRec API",
    description="端到端人脸识别服务 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API 路由
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "name": "DDFD-FaceRec API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """健康检查"""
    import torch
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gpu_available=torch.cuda.is_available(),
        model_loaded=True
    )


@app.post("/api/v1/detect", response_model=DetectResponse, tags=["Detection"])
async def detect(image: UploadFile = File(..., description="人脸图像")):
    """
    人脸检测
    
    - **image**: JPEG/PNG 格式图像
    - 返回检测到的人脸框、置信度、关键点
    """
    start_time = time.time()
    
    try:
        # 读取图像
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # 检测
        pipeline = get_pipeline()
        result = pipeline.detect(image_cv)
        
        # 格式化输出
        faces = []
        for face in result.get('faces', []):
            faces.append({
                'bbox': [float(x) for x in face['bbox']],
                'score': float(face['score']),
                'landmarks': face.get('landmarks'),
            })
        
        inference_time = (time.time() - start_time) * 1000
        
        return DetectResponse(
            success=True,
            message=f"Detected {len(faces)} faces",
            data={
                'faces': faces,
                'count': len(faces),
                'image_size': [image_cv.shape[1], image_cv.shape[0]],
            },
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        logging.error(f"Detection error: {e}")
        return DetectResponse(
            success=False,
            message=str(e),
            inference_time_ms=(time.time() - start_time) * 1000
        )


@app.post("/api/v1/extract", response_model=ExtractResponse, tags=["Recognition"])
async def extract(
    image: UploadFile = File(..., description="人脸图像"),
    bbox: Optional[str] = Query(default=None, description="人脸框 [x1,y1,x2,y2]")
):
    """
    人脸特征提取
    
    - **image**: 人脸图像
    - **bbox**: 可选的人脸框，不提供则自动检测
    - 返回 512 维特征向量
    """
    start_time = time.time()
    
    try:
        # 读取图像
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # 解析 bbox
        bbox_np = None
        if bbox:
            bbox_np = np.array([float(x) for x in bbox.split(',')])
        
        # 提取特征
        pipeline = get_pipeline()
        result = pipeline.extract(image_cv, bbox=bbox_np)
        
        if not result.get('success', False):
            return ExtractResponse(
                success=False,
                message=result.get('error', 'Unknown error'),
                inference_time_ms=(time.time() - start_time) * 1000
            )
        
        inference_time = (time.time() - start_time) * 1000
        
        return ExtractResponse(
            success=True,
            message="Feature extracted successfully",
            data={
                'feature': result['feature'].tolist(),
                'quality': result.get('quality', {}),
                'bbox': result.get('bbox', []),
            },
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        logging.error(f"Extraction error: {e}")
        return ExtractResponse(
            success=False,
            message=str(e),
            inference_time_ms=(time.time() - start_time) * 1000
        )


@app.post("/api/v1/verify", response_model=VerifyResponse, tags=["Matching"])
async def verify(request: VerifyRequest):
    """
    人脸验证 (1:1 比对)
    
    - 比较两个特征向量的相似度
    - 返回是否同一人及置信度
    """
    start_time = time.time()
    
    try:
        pipeline = get_pipeline()
        
        feat1 = np.array(request.feature1, dtype=np.float32)
        feat2 = np.array(request.feature2, dtype=np.float32)
        
        is_same, similarity = pipeline.matcher.verify(
            feat1, feat2, threshold=request.threshold
        )
        
        # 计算置信度
        if is_same:
            confidence = (similarity - request.threshold) / (1 - request.threshold)
        else:
            confidence = (request.threshold - similarity) / request.threshold
        confidence = max(0, min(1, confidence))
        
        inference_time = (time.time() - start_time) * 1000
        
        return VerifyResponse(
            success=True,
            is_same=is_same,
            similarity=similarity,
            confidence=confidence,
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        logging.error(f"Verification error: {e}")
        return VerifyResponse(
            success=False,
            is_same=False,
            similarity=0.0,
            confidence=0.0,
            inference_time_ms=(time.time() - start_time) * 1000
        )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Matching"])
async def search(request: SearchRequest):
    """
    人脸搜索 (1:N 检索)
    
    - 在图库中搜索相似人脸
    - 返回 Top-K 匹配结果
    """
    start_time = time.time()
    
    try:
        pipeline = get_pipeline()
        
        feat = np.array(request.feature, dtype=np.float32)
        
        results = pipeline.matcher.search(
            feat, top_k=request.top_k, threshold=request.threshold
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            success=True,
            results=results,
            count=len(results),
            search_time_ms=inference_time
        )
        
    except Exception as e:
        logging.error(f"Search error: {e}")
        return SearchResponse(
            success=False,
            results=[],
            count=0,
            search_time_ms=(time.time() - start_time) * 1000
        )


@app.post("/api/v1/register", tags=["Gallery"])
async def register(request: RegisterRequest):
    """
    注册人脸到图库
    
    - 将特征向量添加到搜索索引
    """
    try:
        pipeline = get_pipeline()
        
        feat = np.array(request.feature, dtype=np.float32)
        
        result = pipeline.register(
            image=None,  # 已提供特征
            person_id=request.person_id,
            metadata=request.metadata
        )
        
        return {
            'success': result.get('success', False),
            'person_id': request.person_id,
            'message': result.get('message', 'Registration completed')
        }
        
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return {'success': False, 'message': str(e)}


@app.get("/api/v1/stats", tags=["System"])
async def get_stats():
    """获取系统统计信息"""
    pipeline = get_pipeline()
    return pipeline.get_stats()


# ============================================
# 错误处理
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'success': False,
            'error': exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'success': False,
            'error': 'Internal server error'
        }
    )


# ============================================
# 主入口
# ============================================

def main():
    """启动 API 服务"""
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False
    )


if __name__ == "__main__":
    main()
