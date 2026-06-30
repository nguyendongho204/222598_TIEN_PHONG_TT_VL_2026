"""
FastAPI endpoint for dataset training.

This module provides API endpoints for model training and evaluation.
It uses service layer for business logic separation.

Author: Development Team
Version: 1.0.0
"""

import logging
import time
import uuid
import asyncio
from functools import partial
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import TrainingParameters, TrainingResponse, ErrorResponse
from app.services import TrainingOrchestrationService
from app.constants import (
    DEFAULT_TEST_SIZE,
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_FILE_EXTENSIONS,
    STATUS_SUCCESS,
    STATUS_ERROR,
    ERROR_FILE_NOT_PROVIDED,
    ERROR_INVALID_FILE_FORMAT,
    ERROR_FILE_TOO_LARGE,
    ERROR_DATASET_PROCESSING,
)
from app.routers.execution_history import save_execution_history

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api", tags=["training"])

# Initialize service
training_service = TrainingOrchestrationService()


# ============================================================================
# Helper Functions
# ============================================================================

def validate_file(filename: str, file_size: int) -> None:
    """
    Validate uploaded file.
    
    Args:
        filename: Name of uploaded file
        file_size: Size of file in bytes
        
    Raises:
        HTTPException: If validation fails
    """
    if not filename:
        raise HTTPException(
            status_code=400,
            detail=ERROR_FILE_NOT_PROVIDED
        )
    
    # Check file extension
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_FILE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=ERROR_INVALID_FILE_FORMAT
        )
    
    # Check file size
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=ERROR_FILE_TOO_LARGE
        )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/train-dataset",
    response_model=TrainingResponse,
    responses={
        200: {"description": "Training successful"},
        400: {"description": "Bad request", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse},
    }
)
async def train_from_file(
    file: UploadFile = File(
        ...,
        description="CSV file containing training data"
    ),
    test_size: float = Form(
        default=DEFAULT_TEST_SIZE,
        ge=0.05,
        le=0.5,
        description="Test set size (5%-50%)"
    ),
    num_runs: int = Form(
        default=5,
        ge=2,
        le=10,
        description="Number of K-Fold folds (2-10)"
    )
):
    """
    Train JEPA+SVM pipeline on uploaded dataset.
    
    This endpoint accepts a CSV file and training parameters, then runs
    the JEPA+SVM pipeline: self-supervised pre-training, supervised fine-tuning,
    embedding extraction, and SVM classification. Returns accuracy metrics
    with K-Fold Cross Validation.
    
    Args:
        file: CSV file with features (all columns) and target (last column)
        test_size: Proportion of data for testing (default: 0.2)
        num_runs: Number of K-Fold folds (default: 5)
    
    Returns:
        TrainingResponse: Contains dataset info and model metrics
        
    Examples:
        ```bash
        curl -X POST http://localhost:8000/api/train-dataset \\
          -F "file=@iris.csv" \\
          -F "test_size=0.2" \\
          -F "num_runs=5"
        ```
    """
    
    try:
        # Validate file
        logger.info(f"Received training request for file: {file.filename}")
        
        file_content = await file.read()
        validate_file(file.filename, len(file_content))
        
        # Execute training
        logger.info(
            f"Starting training - test_size={test_size}, num_runs={num_runs}"
        )
        
        start_time = time.time()

        # Run training in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        train_fn = partial(
            training_service.execute_training,
            file_content=file_content,
            test_size=test_size,
            num_runs=num_runs,
            dataset_name=file.filename
        )
        result = await loop.run_in_executor(None, train_fn)

        execution_time = time.time() - start_time
        
        logger.info(f"Training completed successfully: {result}")
        
        # Auto-save to execution history
        try:
            execution_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
            save_execution_history(
                execution_id=execution_id,
                algorithm_name="JEPA-SVM",
                file_id=file.filename,
                parameters={
                    "test_size": test_size,
                    "num_runs": num_runs,
                },
                results={k: v for k, v in result.items() if k != "status"},
                execution_time=execution_time,
                status="success"
            )
            logger.info(f"Execution history saved: {execution_id}")
        except Exception as e:
            logger.warning(f"Failed to save execution history: {str(e)}")
        
        return result
        
    except HTTPException as e:
        logger.error(f"Validation error: {e.detail}")
        raise
        
    except ValueError as e:
        error_msg = f"Training error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=400,
            detail=ERROR_DATASET_PROCESSING
        )
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
