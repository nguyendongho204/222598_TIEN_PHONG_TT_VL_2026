"""
Main FastAPI application entry point.

This module initializes the FastAPI application, configures middleware,
registers routers, and sets up application lifecycle management.

Author: Development Team
Version: 1.0.0
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import settings, ensure_directories
from app.routers import base, file_upload, ml_models, execution_history, universal_pipeline
from app.utils.helpers import configure_logging

# Configure logging
configure_logging(
    log_level="INFO",
    log_file="logs/api.log"
)

logger = logging.getLogger(__name__)


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info("=" * 60)
    
    try:
        ensure_directories()
        logger.info("Application directory structure initialized")
    except Exception as e:
        logger.error(f"Failed to initialize directories: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down application...")
    logger.info("=" * 60)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan
)


# ============================================================================
# Middleware
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log incoming HTTP requests.
    
    Args:
        request (Request): HTTP request
        call_next: Next middleware/route
        
    Returns:
        Response: HTTP response
    """
    logger.debug(f"{request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        logger.debug(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        raise


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.
    
    Args:
        request (Request): HTTP request
        exc (Exception): Exception raised
        
    Returns:
        JSONResponse: Error response
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Check server logs for details.",
            "status_code": 500
        }
    )


# ============================================================================
# Router Registration
# ============================================================================

# Register routers
app.include_router(base.router)
app.include_router(file_upload.router)
app.include_router(ml_models.router)
app.include_router(execution_history.router)
app.include_router(universal_pipeline.router)



# ============================================================================  
# Root Endpoint
# ============================================================================  

@app.get("/", summary="API Root")
async def root():
    """
    Root endpoint providing basic API information.

    Returns:
        dict: Welcome message and API metadata
    """
    return {
        "message": f"Welcome to {settings.api_title}",
        "version": settings.api_version,
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on http://0.0.0.0:8000")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
