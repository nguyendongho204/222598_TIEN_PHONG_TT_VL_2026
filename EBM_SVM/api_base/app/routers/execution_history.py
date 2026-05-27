"""
Execution history and result analysis API endpoints.

This module provides endpoints for managing algorithm execution history,
retrieving past runs, and automatically analyzing results.

Author: Development Team
Version: 1.0.0
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel
from app.models.schemas import (
    ExecutionHistory,
    ExecutionHistoryResponse,
    HistoryListResponse,
    ResultAnalysis,
    ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/history", tags=["execution history"])

# History storage directory
HISTORY_DIR = Path("execution_history")


# ============================================================================
# Request Models
# ============================================================================

class SaveHistoryRequest(BaseModel):
    """Request model for saving execution history."""
    execution_id: str
    algorithm_name: str
    file_id: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    execution_time: float
    analysis: Optional[Dict[str, Any]] = None
    status: str = "success"


def ensure_history_directory():
    """Ensure history directory exists."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def save_execution_history(
    execution_id: str,
    algorithm_name: str,
    file_id: str,
    parameters: Dict[str, Any],
    results: Dict[str, Any],
    execution_time: float,
    analysis: Dict[str, Any] = None,
    status: str = "success"
) -> ExecutionHistory:
    """
    Save execution history to disk.
    
    Args:
        execution_id (str): Unique execution ID
        algorithm_name (str): Name of algorithm
        file_id (str): Input file ID
        parameters (Dict[str, Any]): Algorithm parameters
        results (Dict[str, Any]): Execution results
        execution_time (float): Execution duration
        analysis (Dict[str, Any], optional): Result analysis
        status (str): Execution status
        
    Returns:
        ExecutionHistory: Saved history record
    """
    ensure_history_directory()
    
    timestamp = datetime.now().isoformat()
    
    history = ExecutionHistory(
        execution_id=execution_id,
        algorithm_name=algorithm_name,
        file_id=file_id,
        timestamp=timestamp,
        parameters=parameters,
        results=results,
        analysis=analysis,
        execution_time=execution_time,
        status=status
    )
    
    # Save to file
    history_file = HISTORY_DIR / f"{execution_id}.json"
    with open(history_file, "w") as f:
        json.dump(history.dict(), f, indent=2)
    
    logger.info(f"Execution history saved: {execution_id}")
    
    return history


@router.post(
    "/save",
    response_model=ExecutionHistoryResponse,
    summary="Save Execution History",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def save_history(request: SaveHistoryRequest):
    """
    Save algorithm execution history.
    
    Args:
        request (SaveHistoryRequest): Request containing execution details
        
    Returns:
        ExecutionHistoryResponse: Confirmation with saved history
    """
    try:
        history = save_execution_history(
            execution_id=request.execution_id,
            algorithm_name=request.algorithm_name,
            file_id=request.file_id,
            parameters=request.parameters,
            results=request.results,
            execution_time=request.execution_time,
            analysis=request.analysis,
            status=request.status
        )
        
        return ExecutionHistoryResponse(
            message="Execution history saved successfully",
            execution_id=request.execution_id,
            history=history
        )
        
    except Exception as e:
        logger.error(f"Failed to save history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save history: {str(e)}"
        )


@router.get(
    "/all",
    response_model=HistoryListResponse,
    summary="Get All Execution Histories",
    responses={
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_all_histories():
    """
    Get all execution histories sorted by timestamp (newest first).
    
    Returns:
        HistoryListResponse: List of all execution histories
        
    Example:
        GET /api/history/all
        Response: {
            "message": "Retrieved execution histories",
            "total": 5,
            "histories": [...]
        }
    """
    try:
        ensure_history_directory()
        
        histories = []
        if HISTORY_DIR.exists():
            for file in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
                with open(file, "r") as f:
                    history_data = json.load(f)
                    histories.append(ExecutionHistory(**history_data))
        
        return HistoryListResponse(
            message="Retrieved execution histories",
            total=len(histories),
            histories=histories
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve histories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve histories: {str(e)}"
        )


@router.get(
    "/{execution_id}",
    response_model=ExecutionHistoryResponse,
    summary="Get Execution History by ID",
    responses={
        404: {"model": ErrorResponse, "description": "History not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_history(execution_id: str):
    """
    Get specific execution history by ID.
    
    Args:
        execution_id (str): Execution ID to retrieve
        
    Returns:
        ExecutionHistoryResponse: Requested history record
        
    Raises:
        HTTPException: If history not found
        
    Example:
        GET /api/history/20240406_120000_a1b2c3d4
        Response: {
            "message": "Execution history retrieved",
            "execution_id": "20240406_120000_a1b2c3d4",
            "history": {...}
        }
    """
    try:
        history_file = HISTORY_DIR / f"{execution_id}.json"
        
        if not history_file.exists():
            raise FileNotFoundError(f"History not found: {execution_id}")
        
        with open(history_file, "r") as f:
            history_data = json.load(f)
        
        history = ExecutionHistory(**history_data)
        
        return ExecutionHistoryResponse(
            message="Execution history retrieved",
            execution_id=execution_id,
            history=history
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution history not found: {execution_id}"
        )
    except Exception as e:
        logger.error(f"Failed to retrieve history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get(
    "/algorithm/{algorithm_name}",
    response_model=HistoryListResponse,
    summary="Get Histories by Algorithm",
    responses={
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_histories_by_algorithm(algorithm_name: str):
    """
    Get all execution histories for a specific algorithm.
    
    Args:
        algorithm_name (str): Algorithm name to filter by
        
    Returns:
        HistoryListResponse: Filtered history records
        
    Example:
        GET /api/history/algorithm/SVM
        Response: {
            "message": "Retrieved SVM histories",
            "total": 3,
            "histories": [...]
        }
    """
    try:
        ensure_history_directory()
        
        histories = []
        if HISTORY_DIR.exists():
            for file in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
                with open(file, "r") as f:
                    history_data = json.load(f)
                    if history_data.get("algorithm_name") == algorithm_name:
                        histories.append(ExecutionHistory(**history_data))
        
        return HistoryListResponse(
            message=f"Retrieved {algorithm_name} execution histories",
            total=len(histories),
            histories=histories
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve histories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve histories: {str(e)}"
        )


@router.delete(
    "/{execution_id}",
    response_model=Dict[str, str],
    summary="Delete Execution History",
    responses={
        404: {"model": ErrorResponse, "description": "History not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def delete_history(execution_id: str):
    """
    Delete an execution history record.
    
    Args:
        execution_id (str): Execution ID to delete
        
    Returns:
        Dict[str, str]: Deletion confirmation
        
    Raises:
        HTTPException: If history not found
        
    Example:
        DELETE /api/history/20240406_120000_a1b2c3d4
        Response: {"message": "History deleted successfully"}
    """
    try:
        history_file = HISTORY_DIR / f"{execution_id}.json"
        
        if not history_file.exists():
            raise FileNotFoundError(f"History not found: {execution_id}")
        
        history_file.unlink()
        
        logger.info(f"Execution history deleted: {execution_id}")
        
        return {"message": "History deleted successfully"}
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution history not found: {execution_id}"
        )
    except Exception as e:
        logger.error(f"Failed to delete history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete history: {str(e)}"
        )


def analyze_results(
    algorithm_name: str,
    results: Dict[str, Any]
) -> ResultAnalysis:
    """
    Automatically analyze algorithm results.
    
    Args:
        algorithm_name (str): Name of algorithm
        results (Dict[str, Any]): Execution results
        
    Returns:
        ResultAnalysis: Analysis results with recommendations
    """
    key_findings = []
    recommendations = []
    best_metric = ""
    worst_metric = ""
    
    # Extract accuracy or main metric
    accuracy = results.get("accuracy", results.get("svm_accuracy_embeddings", 0))
    if isinstance(accuracy, float):
        accuracy_score = min(accuracy, 1.0) if accuracy > 1 else accuracy
    else:
        accuracy_score = 0
    
    # Generate findings based on accuracy
    if accuracy_score >= 0.95:
        key_findings.append("Mô hình hoạt động xuất sắc vói độ chính xác rất cao")
        best_metric = "accuracy"
    elif accuracy_score >= 0.85:
        key_findings.append("Mô hình hoạt động tốt với độ chính xác khá")
        best_metric = "accuracy"
    elif accuracy_score >= 0.75:
        key_findings.append("Hiệu năng mô hình ở mức trung bình; cần kỹ thuật trích xuất đặc trưng")
        worst_metric = "accuracy"
    else:
        key_findings.append("Mô hình hoạt động kém; khuyến nghị tối ưu hóa đặc trưng")
        worst_metric = "accuracy"

    # Check for imbalanced data or other issues
    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        if isinstance(cm, list) and len(cm) > 0:
            key_findings.append(f"Mô hình được đánh giá trên {len(cm)} lớp (classes)")

    # Add specific algorithm insights
    if algorithm_name == "EBM-SVM" or algorithm_name == "both":
        if "svm_accuracy_original" in results and "svm_accuracy_embeddings" in results:
            orig = results.get("svm_accuracy_original", 0)
            emb = results.get("svm_accuracy_embeddings", 0)
            if emb > orig:
                key_findings.append("Các vector nhúng của EBM đã giúp cải thiện độ chính xác đáng kể")
                recommendations.append("Tiếp tục sử dụng vector nhúng EBM cho việc trích xuất đặc trưng")
            else:
                recommendations.append("Có thể cân nhắc dùng đặc trưng gốc nếu EBM không hiệu quả hơn")
        elif "ebm" in results and "svm" in results:
            orig = results.get("svm", {}).get("accuracy", 0)
            emb = results.get("ebm", {}).get("accuracy", 0)
            if emb > orig:
                key_findings.append("Mô hình kết hợp EBM-SVM đã phân loại chính xác hơn thuật toán SVM thông thường")
                recommendations.append("Sự biểu diễn không gian thông qua EBM hỗ trợ RBF kernel phân lớp tốt hơn")
            elif emb == orig:
                key_findings.append("Cả SVM gốc và EBM-SVM kết hợp đều cho ra kết quả tương đương nhau")
                recommendations.append("Dữ liệu này có thể đã đủ tốt để SVM phân loại mà không cần EBM")
            else:
                recommendations.append("Nên dùng đặc trưng gốc do EBM có thể đã làm mất đi đặc trưng quan trọng")

    # General recommendations
    if accuracy_score < 0.80:
        recommendations.append("Cân nhắc thu thập thêm dữ liệu huấn luyện để dạy mô hình")
        recommendations.append("Thử một số phương pháp lọc đặc trưng (Feature Selection)")
        recommendations.append("Thử nghiệm với các siêu tham số (Hyperparameters) khác nhau")

    if accuracy_score >= 0.90:
        recommendations.append("Mô hình hoạt động rất tốt; có thể triển khai lên môi trường sản phẩm")
        recommendations.append("Nên ưu tiên dùng phiên bản mô hình này để suy luận dữ liệu mới")

    return ResultAnalysis(
        accuracy_score=accuracy_score,
        best_metric=best_metric or "accuracy",
        worst_metric=worst_metric or "N/A",
        key_findings=key_findings,
        recommendations=recommendations,
        comparison=None
    )


@router.post(
    "/analyze",
    response_model=ExecutionHistoryResponse,
    summary="Analyze Execution Results",
    responses={
        404: {"model": ErrorResponse, "description": "History not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def analyze_execution(execution_id: str):
    """
    Perform automated analysis on execution results and save analysis.
    
    Args:
        execution_id (str): Execution ID to analyze
        
    Returns:
        ExecutionHistoryResponse: History with analysis attached
        
    Example:
        POST /api/history/analyze?execution_id=20240406_120000_a1b2c3d4
        Response: {
            "message": "Analysis completed",
            "execution_id": "20240406_120000_a1b2c3d4",
            "history": {...with analysis field populated...}
        }
    """
    try:
        history_file = HISTORY_DIR / f"{execution_id}.json"
        
        if not history_file.exists():
            raise FileNotFoundError(f"History not found: {execution_id}")
        
        with open(history_file, "r") as f:
            history_data = json.load(f)
        
        history = ExecutionHistory(**history_data)
        
        # Perform analysis
        analysis = analyze_results(history.algorithm_name, history.results)
        
        # Update history with analysis
        history.analysis = analysis.dict()
        
        # Save updated history
        with open(history_file, "w") as f:
            json.dump(history.dict(), f, indent=2)
        
        logger.info(f"Analysis completed for execution: {execution_id}")
        
        return ExecutionHistoryResponse(
            message="Analysis completed successfully",
            execution_id=execution_id,
            history=history
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution history not found: {execution_id}"
        )
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )
