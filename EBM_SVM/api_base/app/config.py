"""
Configuration module for EBM-SVM API application.

This module manages all configuration settings loaded from environment variables
and provides centralized access to project configuration including API settings,
ML model parameters, and file paths.

Author: Development Team
Version: 1.0.0
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    This class uses Pydantic's BaseSettings to validate and manage
    all configuration parameters for the application, ensuring type safety
    and default value management.
    
    Attributes:
        api_title (str): Title of the FastAPI application
        api_description (str): Description of the API
        api_version (str): Version of the API
        debug (bool): Debug mode flag
        allowed_origins (list): CORS allowed origins
        
        upload_dir (str): Directory for temporary file uploads
        download_dir (str): Directory for file downloads
        max_upload_size (int): Maximum file upload size in bytes
        
        ebm_epochs (int): Number of training epochs for EBM
        ebm_lr (float): Learning rate for EBM training
        ebm_hidden_dim (int): Hidden dimension for EBM
        ebm_embedding_dim (int): Embedding dimension for EBM
        ebm_noise_scale (float): Noise scale for EBM training
        
        svm_kernel (str): SVM kernel type
        svm_c (float): SVM regularization parameter
        test_size (float): Test data split ratio
        random_seed (int): Random seed for reproducibility
    """
    
    # API Configuration
    api_title: str = Field(default="EBM-SVM Demo API", description="API Title")
    api_description: str = Field(default="API for EBM-SVM integration and machine learning tasks",
                                 description="API Description")
    api_version: str = Field(default="1.0.0", description="API Version")
    debug: bool = Field(default=True, description="Debug mode")
    
    # CORS Configuration
    allowed_origins: list = Field(default=["http://localhost:3000", "http://localhost:3001", "http://localhost:8080"],
                                  description="CORS allowed origins")
    
    # File Upload Configuration
    upload_dir: str = Field(default="utils/upload_temp", description="Upload directory")
    download_dir: str = Field(default="utils/download", description="Download directory")
    max_upload_size: int = Field(default=52428800, description="Maximum upload size (50MB)")
    allowed_file_extensions: list = Field(default=[".csv", ".xlsx", ".txt"],
                                          description="Allowed file extensions")
    
    # EBM Model Configuration
    ebm_epochs: int = Field(default=3000, description="EBM training epochs")
    ebm_lr: float = Field(default=0.0001, description="EBM learning rate")
    ebm_hidden_dim: int = Field(default=512, description="EBM hidden dimension")
    ebm_embedding_dim: int = Field(default=64, description="EBM embedding dimension")
    ebm_noise_scale: float = Field(default=2.0, description="EBM noise scale")
    
    # SVM Configuration
    svm_kernel: str = Field(default="rbf", description="SVM kernel type")
    svm_c: float = Field(default=1.0, description="SVM C parameter")
    
    # Data Processing Configuration
    test_size: float = Field(default=0.2, description="Test data split ratio")
    random_seed: int = Field(default=42, description="Random seed")
    
    class Config:
        """Pydantic configuration for Settings."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings: Application settings instance with values from .env or defaults
    """
    return Settings()


def ensure_directories() -> None:
    """
    Ensure all required directories exist.
    
    Creates upload_temp and download directories if they don't exist.
    This function should be called during application startup.
    
    Raises:
        OSError: If directories cannot be created due to permission errors
    """
    settings = get_settings()
    
    try:
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.download_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create required directories: {str(e)}")


# Export settings instance for easy access
settings = get_settings()
