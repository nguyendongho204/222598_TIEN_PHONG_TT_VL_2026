"""
File upload and management API endpoints.

This module handles CSV file uploads, file management, and data validation
for the machine learning pipeline.

Author: Development Team
Version: 1.0.0
"""

import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.models.schemas import FileUploadResponse, ErrorResponse
from app.utils.helpers import FileManager
from app.config import settings
import pandas as pd
import io

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/files", tags=["files"])

# Initialize file manager
file_manager = FileManager(
    upload_dir=settings.upload_dir,
    download_dir=settings.download_dir,
    allowed_extensions=settings.allowed_file_extensions,
    max_file_size=settings.max_upload_size
)


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    summary="Upload CSV File",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV file for processing.
    
    The uploaded file is temporarily stored and can be used for training
    machine learning models. File must be in CSV format.
    
    Args:
        file (UploadFile): CSV file to upload
        
    Returns:
        FileUploadResponse: Upload confirmation with file metadata
        
    Raises:
        HTTPException: If file is invalid or too large
        
    Example:
        POST /api/files/upload
        Body: multipart/form-data with file
        Response: {
            "message": "Data uploaded successfully",
            "filename": "data.csv",
            "columns": ["feature1", "feature2", "label"],
            "rows": 100,
            "file_id": "20240406_120000_a1b2c3d4"
        }
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Validate file
        file_manager.validate_file(file.filename, len(contents))
        
        # Parse CSV to validate and get metadata
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Save file
        file_id = file_manager.save_upload(file.filename, contents)
        
        logger.info(f"File uploaded successfully: {file_id}")
        
        return FileUploadResponse(
            message="Data uploaded successfully",
            filename=file.filename,
            columns=list(df.columns),
            rows=len(df),
            file_id=file_id
        )
        
    except ValueError as e:
        logger.warning(f"File validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )


@router.get(
    "/info/{file_id}",
    summary="Get File Information",
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
    }
)
async def get_file_info(file_id: str):
    """
    Get information about an uploaded file.
    
    Args:
        file_id (str): Unique file identifier
        
    Returns:
        dict: File metadata (shape, columns, samples)
        
    Raises:
        HTTPException: If file not found
        
    Example:
        GET /api/files/info/20240406_120000_a1b2c3d4
        Response: {
            "file_id": "20240406_120000_a1b2c3d4",
            "shape": [100, 5],
            "columns": ["feature1", "feature2", "label"],
            "rows": 100
        }
    """
    try:
        file_path = file_manager.get_upload_path(file_id)
        df = pd.read_csv(file_path)
        
        return {
            "file_id": file_id,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "rows": len(df)
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_id}"
        )
    except Exception as e:
        logger.error(f"Failed to get file info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading file: {str(e)}"
        )
