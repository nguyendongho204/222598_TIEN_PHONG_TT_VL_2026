"""
Data models and schemas for the EBM-SVM API application.

This module defines Pydantic models for request/response validation,
database models, and data transfer objects (DTOs) to ensure type safety
and API documentation consistency.

Author: Development Team
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class DatasetType(str, Enum):
    """Enumeration for supported dataset types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# ============================================================================
# Request/Response Models
# ============================================================================

class HealthCheckResponse(BaseModel):
    """
    Health check response model.
    
    Attributes:
        status (str): Current status of the API
        version (str): API version
    """
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")


class FileUploadResponse(BaseModel):
    """
    Response model for file upload endpoint.
    
    Attributes:
        message (str): Upload success message
        filename (str): Uploaded filename
        columns (List[str]): Column names in the dataset
        rows (int): Number of rows in the dataset
        file_id (str): Unique file identifier
    """
    message: str = Field(..., description="Success message")
    filename: str = Field(..., description="Uploaded filename")
    columns: List[str] = Field(..., description="Dataset columns")
    rows: int = Field(..., description="Number of rows")
    file_id: str = Field(..., description="File identifier")


class TrainingRequest(BaseModel):
    """
    Request model for model training.
    
    Attributes:
        file_id (str): ID of uploaded file to train on
        test_size (float): Test data split ratio (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility
    """
    file_id: str = Field(..., description="Uploaded file ID", min_length=1)
    test_size: float = Field(default=0.2, ge=0.0, le=1.0, description="Test size ratio")
    random_seed: int = Field(default=42, description="Random seed")
    
    @validator('test_size')
    def validate_test_size(cls, v):
        """Validate test size is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")
        return v


class TrainingResponse(BaseModel):
    """
    Response model for training completion.
    
    Attributes:
        message (str): Training completion message
        training_id (str): Training job identifier
        svm_accuracy_original (float): SVM accuracy on original features
        svm_accuracy_embeddings (float): SVM accuracy on EBM embeddings
        training_time (float): Training duration in seconds
    """
    message: str = Field(..., description="Training message")
    training_id: str = Field(..., description="Training identifier")
    svm_accuracy_original: float = Field(..., description="Original SVM accuracy", ge=0.0, le=1.0)
    svm_accuracy_embeddings: float = Field(..., description="EBM embeddings SVM accuracy", ge=0.0, le=1.0)
    training_time: float = Field(..., description="Training duration in seconds")


class PredictionRequest(BaseModel):
    """
    Request model for making predictions.
    
    Attributes:
        training_id (str): ID of trained model
        file_id (str): ID of file with features to predict
    """
    training_id: str = Field(..., description="Training ID")
    file_id: str = Field(..., description="File ID for predictions")


class PredictionResponse(BaseModel):
    """
    Response model for predictions.
    
    Attributes:
        training_id (str): ID of model used
        predictions (List[Any]): List of predicted labels
        confidence (Optional[List[float]]): Prediction confidence scores
    """
    training_id: str = Field(..., description="Training ID")
    predictions: List[Any] = Field(..., description="Predicted labels")
    confidence: Optional[List[float]] = Field(None, description="Confidence scores")


class ClassificationReport(BaseModel):
    """
    Classification metrics report.
    
    Attributes:
        precision (float): Precision score
        recall (float): Recall score
        f1_score (float): F1 score
        support (int): Number of samples
    """
    precision: float = Field(..., description="Precision", ge=0.0, le=1.0)
    recall: float = Field(..., description="Recall", ge=0.0, le=1.0)
    f1_score: float = Field(..., description="F1 Score", ge=0.0, le=1.0)
    support: int = Field(..., description="Support count", ge=0)


class ModelEvaluationResponse(BaseModel):
    """
    Model evaluation and metrics response.
    
    Attributes:
        accuracy (float): Overall accuracy
        report (Dict[str, ClassificationReport]): Per-class metrics
        confusion_matrix (List[List[int]]): Confusion matrix
    """
    accuracy: float = Field(..., description="Model accuracy", ge=0.0, le=1.0)
    report: Dict[str, Any] = Field(..., description="Classification report")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Attributes:
        error (str): Error type/category
        message (str): Detailed error message
        status_code (int): HTTP status code
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")


# ============================================================================
# Configuration Models
# ============================================================================

class EBMConfig(BaseModel):
    """
    EBM model configuration.
    
    Attributes:
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        hidden_dim (int): Hidden layer dimension
        embedding_dim (int): Embedding dimension
        noise_scale (float): Noise scale for training
    """
    epochs: int = Field(default=200, ge=1, description="Training epochs")
    learning_rate: float = Field(default=0.0005, gt=0.0, description="Learning rate")
    hidden_dim: int = Field(default=256, ge=1, description="Hidden dimension")
    embedding_dim: int = Field(default=64, ge=1, description="Embedding dimension")
    noise_scale: float = Field(default=0.5, ge=0.0, description="Noise scale")


class SVMConfig(BaseModel):
    """
    SVM model configuration.
    
    Attributes:
        kernel (str): Kernel type (linear, rbf, poly, sigmoid)
        c (float): Regularization parameter
    """
    kernel: str = Field(default="rbf", description="SVM kernel type")
    c: float = Field(default=1.0, gt=0.0, description="Regularization parameter C")


class DataProcessingConfig(BaseModel):
    """
    Data processing configuration.
    
    Attributes:
        test_size (float): Test data split ratio
        random_seed (int): Random seed for reproducibility
        normalize (bool): Whether to normalize features
    """
    test_size: float = Field(default=0.2, ge=0.0, le=1.0, description="Test size")
    random_seed: int = Field(default=42, description="Random seed")
    normalize: bool = Field(default=True, description="Normalize features")


# ============================================================================
# Execution History Models
# ============================================================================

class ExecutionHistory(BaseModel):
    """
    Execution history record for algorithm runs.
    
    Attributes:
        execution_id (str): Unique execution identifier
        algorithm_name (str): Name of algorithm executed (e.g., 'SVM', 'EBM-SVM')
        file_id (str): ID of data file used
        timestamp (str): Execution timestamp (ISO format)
        parameters (Dict[str, Any]): Algorithm parameters
        results (Dict[str, Any]): Execution results and metrics
        analysis (Dict[str, Any]): Automatic result analysis
        execution_time (float): Execution duration in seconds
        status (str): Execution status (success, error)
    """
    execution_id: str = Field(..., description="Unique execution ID")
    algorithm_name: str = Field(..., description="Algorithm name")
    file_id: str = Field(..., description="Input file ID")
    timestamp: str = Field(..., description="Execution timestamp")
    parameters: Dict[str, Any] = Field(..., description="Algorithm parameters")
    results: Dict[str, Any] = Field(..., description="Execution results")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Result analysis")
    execution_time: float = Field(..., description="Execution time in seconds")
    status: str = Field(default="success", description="Execution status")


class ExecutionHistoryResponse(BaseModel):
    """
    Response for execution history operations.
    
    Attributes:
        message (str): Status message
        execution_id (str): Execution ID
        history (Optional[ExecutionHistory]): Full history record
    """
    message: str = Field(..., description="Response message")
    execution_id: str = Field(..., description="Execution ID")
    history: Optional[ExecutionHistory] = Field(None, description="History record")


class HistoryListResponse(BaseModel):
    """
    Response for listing all execution histories.
    
    Attributes:
        message (str): Status message
        total (int): Total number of records
        histories (List[ExecutionHistory]): List of history records
    """
    message: str = Field(..., description="Response message")
    total: int = Field(..., description="Total records")
    histories: List[ExecutionHistory] = Field(..., description="History records")


class ResultAnalysis(BaseModel):
    """
    Automatic analysis of algorithm results.
    
    Attributes:
        accuracy_score (float): Model accuracy
        best_metric (str): Best performing metric
        worst_metric (str): Worst performing metric
        key_findings (List[str]): Key findings from results
        recommendations (List[str]): Recommendations for improvement
        comparison (Dict[str, str]): Comparison with baseline if available
    """
    accuracy_score: float = Field(..., description="Accuracy score", ge=0.0, le=1.0)
    best_metric: str = Field(..., description="Best metric")
    worst_metric: str = Field(..., description="Worst metric")
    key_findings: List[str] = Field(..., description="Key findings")
    recommendations: List[str] = Field(..., description="Recommendations")
    comparison: Optional[Dict[str, Any]] = Field(None, description="Comparisons")
