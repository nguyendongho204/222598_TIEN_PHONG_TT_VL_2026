from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class TrainingParameters(BaseModel):
    """Tham số huấn luyện từ client"""
    test_size: float = 0.2
    num_runs: int = 5


class TrainingResponse(BaseModel):
    """Kết quả trả về sau khi huấn luyện"""
    status: str
    dataset_name: str
    samples: int
    features: int
    classes: int
    accuracy: float
    accuracy_std: float
    per_fold_accuracies: List[float]
    embedding_dim: int
    precision: List[float]
    recall: List[float]
    f1_score: List[float]
    support: List[int]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    per_class_accuracy: List[float]


class ErrorResponse(BaseModel):
    """Phản hồi lỗi"""
    detail: str
