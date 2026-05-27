"""
Utility functions and helper classes for the EBM-SVM API.

This module provides common utilities such as file handling, data validation,
and logging configuration.

Author: Development Team
Version: 1.0.0
"""

import os
import logging
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


# ============================================================================
# File Management Utilities
# ============================================================================

class FileManager:
    """
    Service for managing uploaded and processed files.
    
    Handles file storage, retrieval, validation, and cleanup.
    
    Attributes:
        upload_dir (str): Directory for temporary uploads
        download_dir (str): Directory for downloads
        allowed_extensions (List[str]): Allowed file extensions
        max_file_size (int): Maximum file size in bytes
    """
    
    def __init__(
        self,
        upload_dir: str = "utils/upload_temp",
        download_dir: str = "utils/download",
        allowed_extensions: List[str] = None,
        max_file_size: int = 52428800  # 50MB
    ):
        """
        Initialize FileManager.
        
        Args:
            upload_dir (str): Upload directory path
            download_dir (str): Download directory path
            allowed_extensions (List[str]): List of allowed file extensions
            max_file_size (int): Maximum file size in bytes
        """
        self.upload_dir = upload_dir
        self.download_dir = download_dir
        self.allowed_extensions = allowed_extensions or [".csv", ".xlsx", ".txt"]
        self.max_file_size = max_file_size
        
        # Create directories
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_file_id(self, filename: str) -> str:
        """
        Generate unique file ID.
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Unique file identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"
    
    def validate_file(self, filename: str, file_size: int) -> bool:
        """
        Validate uploaded file.
        
        Args:
            filename (str): Name of the file
            file_size (int): Size of the file in bytes
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If file is invalid
        """
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_extensions:
            raise ValueError(f"File extension {ext} not allowed. Allowed: {self.allowed_extensions}")
        
        # Check size
        if file_size > self.max_file_size:
            raise ValueError(f"File size {file_size} exceeds maximum {self.max_file_size}")
        
        return True
    
    def save_upload(self, filename: str, contents: bytes) -> str:
        """
        Save uploaded file.
        
        Args:
            filename (str): Original filename
            contents (bytes): File contents
            
        Returns:
            str: File ID of saved file
            
        Raises:
            ValueError: If file validation fails
            IOError: If file saving fails
        """
        try:
            # Validate
            self.validate_file(filename, len(contents))
            
            # Generate ID and path
            file_id = self.generate_file_id(filename)
            file_path = Path(self.upload_dir) / f"{file_id}.csv"  # Save as CSV
            
            # Save
            with open(file_path, "wb") as f:
                f.write(contents)
            
            logger.info(f"File saved: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"File save failed: {str(e)}")
            raise
    
    def get_upload_path(self, file_id: str) -> str:
        """
        Get path to uploaded file.
        
        Args:
            file_id (str): File identifier
            
        Returns:
            str: Full path to file
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(self.upload_dir) / f"{file_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_id}")
        return str(file_path)
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """
        Remove files older than specified days.
        
        Args:
            days (int): Age threshold in days
            
        Returns:
            int: Number of files removed
        """
        removed_count = 0
        threshold = datetime.now().timestamp() - (days * 24 * 3600)
        
        for file_path in Path(self.upload_dir).glob("*.csv"):
            if file_path.stat().st_mtime < threshold:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove file {file_path}: {str(e)}")
        
        logger.info(f"Cleaned up {removed_count} old files")
        return removed_count


# ============================================================================
# Logging Configuration
# ============================================================================

def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure application logging.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): File to write logs to. If None, logs to console only
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)


# ============================================================================
# Data Validation Utilities
# ============================================================================

class DataValidator:
    """
    Utilities for data validation.
    
    Provides functions to validate data shapes, types, and values.
    """
    
    @staticmethod
    def validate_features_shape(X: object, expected_cols: Optional[int] = None) -> bool:
        """
        Validate feature array shape.
        
        Args:
            X: Feature array
            expected_cols (Optional[int]): Expected number of columns
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If invalid
        """
        try:
            if len(X.shape) != 2:
                raise ValueError(f"Expected 2D array, got {len(X.shape)}D")
            
            if expected_cols and X.shape[1] != expected_cols:
                raise ValueError(f"Expected {expected_cols} columns, got {X.shape[1]}")
            
            return True
        except Exception as e:
            logger.error(f"Shape validation failed: {str(e)}")
            raise
    
    @staticmethod
    def validate_labels(y: object) -> bool:
        """
        Validate label array.
        
        Args:
            y: Label array
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If invalid
        """
        try:
            if len(y.shape) != 1:
                raise ValueError(f"Labels must be 1D, got {len(y.shape)}D")
            
            if len(y) == 0:
                raise ValueError("Labels cannot be empty")
            
            return True
        except Exception as e:
            logger.error(f"Label validation failed: {str(e)}")
            raise
