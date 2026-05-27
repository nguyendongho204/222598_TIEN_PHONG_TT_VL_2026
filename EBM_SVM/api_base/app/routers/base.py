"""
Basic API endpoints (health check, info).

This module provides fundamental API endpoints for monitoring and verifying
the API service status.

Author: Development Team
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import HealthCheckResponse
from app.config import settings

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthCheckResponse, summary="Health Check")
async def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        HealthCheckResponse: Status and version information
        
    Example:
        GET /api/health
        Response: {"status": "ok", "version": "1.0.0"}
    """
    return HealthCheckResponse(
        status="ok",
        version=settings.api_version
    )


@router.get("/info", summary="API Information")
async def get_info():
    """
    Get API information and configuration.
    
    Returns:
        dict: API metadata and configuration
        
    Example:
        GET /api/info
        Response: {"title": "EBM-SVM Demo API", "version": "1.0.0", ...}
    """
    return {
        "title": settings.api_title,
        "description": settings.api_description,
        "version": settings.api_version,
        "debug": settings.debug,
        "ebm_config": {
            "epochs": settings.ebm_epochs,
            "learning_rate": settings.ebm_lr,
            "hidden_dim": settings.ebm_hidden_dim,
            "embedding_dim": settings.ebm_embedding_dim,
        },
        "svm_config": {
            "kernel": settings.svm_kernel,
            "c": settings.svm_c,
        }
    }
