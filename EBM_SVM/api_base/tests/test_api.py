"""
Unit tests for the EBM-SVM API application.

This module contains pytest test cases for API endpoints, ML services,
and utility functions.

Author: Development Team
Version: 1.0.0
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
import tempfile
import pandas as pd
import os


# ============================================================================
# Test Client Setup
# ============================================================================

@pytest.fixture
def client():
    """Provide FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_csv():
    """Create a sample CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create simple dataset
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0] * 5,
            'feature2': [2.0, 3.0, 4.0, 5.0] * 5,
            'label': [0, 1, 0, 1] * 5
        })
        df.to_csv(f.name, index=False)
        yield f.name
        # Cleanup
        os.unlink(f.name)


# ============================================================================
# Health Check Tests
# ============================================================================

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_api_info(client):
    """Test API info endpoint."""
    response = client.get("/api/info")
    
    assert response.status_code == 200
    data = response.json()
    assert data["title"]
    assert "ebm_config" in data
    assert "svm_config" in data


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs_url" in data


# ============================================================================
# File Upload Tests
# ============================================================================

def test_file_upload_csv(client, sample_csv):
    """Test CSV file upload."""
    with open(sample_csv, 'rb') as f:
        response = client.post(
            "/api/files/upload",
            files={"file": ("test.csv", f, "text/csv")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["rows"] == 20
    assert "label" in data["columns"]


def test_file_upload_no_file(client):
    """Test file upload without file."""
    response = client.post("/api/files/upload", files={})
    
    assert response.status_code == 400


def test_invalid_file_extension(client):
    """Test uploading invalid file type."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("invalid")
        f.flush()
        
        with open(f.name, 'rb') as upload_file:
            response = client.post(
                "/api/files/upload",
                files={"file": (f.name, upload_file, "text/plain")}
            )
        
        assert response.status_code == 400
        os.unlink(f.name)


# ============================================================================
# ML Models Tests
# ============================================================================

@pytest.mark.asyncio
async def test_train_model_no_file(client):
    """Test training with non-existent file."""
    response = client.post(
        "/api/ml/train",
        json={
            "file_id": "nonexistent_id",
            "test_size": 0.2,
            "random_seed": 42
        }
    )
    
    assert response.status_code == 404


def test_train_model_invalid_params(client):
    """Test training with invalid parameters."""
    response = client.post(
        "/api/ml/train",
        json={
            "file_id": "valid_id",
            "test_size": 1.5,  # Invalid: > 1.0
            "random_seed": 42
        }
    )
    
    assert response.status_code == 422  # Validation error


# ============================================================================
# Configuration Tests
# ============================================================================

def test_settings_loading():
    """Test configuration loading."""
    from app.config import settings
    
    assert settings.api_title
    assert settings.api_version
    assert settings.ebm_epochs > 0
    assert settings.ebm_lr > 0
    assert 0 < settings.test_size < 1


# ============================================================================
# Data Processor Tests
# ============================================================================

def test_data_processor_load_csv(sample_csv):
    """Test CSV loading."""
    from ml_models.services.pipeline import DataProcessor
    
    processor = DataProcessor()
    df = processor.load_csv(sample_csv)
    
    assert df.shape[0] == 20
    assert 'label' in df.columns


def test_data_processor_extract_features(sample_csv):
    """Test feature extraction."""
    from ml_models.services.pipeline import DataProcessor
    
    processor = DataProcessor()
    df = processor.load_csv(sample_csv)
    X, y = processor.extract_features_and_labels(df)
    
    assert X.shape[0] == 20
    assert X.shape[1] == 2  # feature1, feature2
    assert len(y) == 20
    assert set(y) == {0, 1}


def test_data_processor_train_test_split(sample_csv):
    """Test train/test split."""
    from ml_models.services.pipeline import DataProcessor
    
    processor = DataProcessor()
    df = processor.load_csv(sample_csv)
    X, y = processor.extract_features_and_labels(df)
    
    X_train, X_test, y_train, y_test = processor.train_test_split_data(
        X, y, test_size=0.2, random_state=42
    )
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == int(len(X) * 0.2)


# ============================================================================
# File Manager Tests
# ============================================================================

def test_file_manager_validation():
    """Test file validation."""
    from app.utils.helpers import FileManager
    
    fm = FileManager()
    
    # Valid file
    assert fm.validate_file("data.csv", 1000)
    
    # Invalid extension
    with pytest.raises(ValueError):
        fm.validate_file("data.txt", 1000)
    
    # File too large
    with pytest.raises(ValueError):
        fm.validate_file("data.csv", 100000000)  # 100MB


def test_file_manager_id_generation():
    """Test file ID generation."""
    from app.utils.helpers import FileManager
    
    fm = FileManager()
    
    file_id1 = fm.generate_file_id("test1.csv")
    file_id2 = fm.generate_file_id("test2.csv")
    
    assert file_id1 != file_id2
    assert len(file_id1) > 0


# ============================================================================
# Model Tests
# ============================================================================

def test_ebm_encoder_initialization():
    """Test EBM encoder initialization."""
    from ml_models.ebm_svm import EBMEncoder
    import torch
    
    ebm = EBMEncoder(input_dim=10, hidden_dim=64, embedding_dim=32)
    
    assert ebm.input_dim == 10
    assert ebm.hidden_dim == 64
    assert ebm.embedding_dim == 32


def test_ebm_encoder_forward_pass():
    """Test EBM encoder forward pass."""
    from ml_models.ebm_svm import EBMEncoder
    import torch
    
    ebm = EBMEncoder(input_dim=10, hidden_dim=64, embedding_dim=32)
    x = torch.randn(8, 10)
    
    embeddings, energy = ebm(x)
    
    assert embeddings.shape == (8, 32)
    assert energy.shape == (8, 1)


def test_svm_classifier_initialization():
    """Test SVM classifier initialization."""
    from ml_models.ebm_svm import SVMClassifier
    
    svm = SVMClassifier(kernel='rbf', C=1.0)
    
    assert svm.kernel == 'rbf'
    assert svm.C == 1.0
    assert not svm._is_fitted


def test_svm_classifier_training():
    """Test SVM classifier training."""
    from ml_models.ebm_svm import SVMClassifier
    import numpy as np
    
    X_train = np.random.randn(20, 5)
    y_train = np.array([0, 1] * 10)
    
    svm = SVMClassifier(kernel='rbf')
    svm.fit(X_train, y_train)
    
    assert svm._is_fitted


def test_svm_classifier_prediction():
    """Test SVM classifier prediction."""
    from ml_models.ebm_svm import SVMClassifier
    import numpy as np
    
    X_train = np.random.randn(20, 5)
    y_train = np.array([0, 1] * 10)
    X_test = np.random.randn(5, 5)
    
    svm = SVMClassifier(kernel='rbf')
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    
    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.slow
def test_pipeline_training_workflow(sample_csv):
    """Integration test for complete training workflow."""
    from ml_models.services.pipeline import EBMSVMPipeline
    
    pipeline = EBMSVMPipeline()
    
    result = pipeline.train(
        file_path=sample_csv,
        test_size=0.2,
        random_state=42,
        ebm_config={
            "epochs": 10,  # Reduced for testing
            "learning_rate": 0.0005,
            "hidden_dim": 32,
            "embedding_dim": 16,
            "noise_scale": 0.5
        }
    )
    
    assert "training_id" in result
    assert "metadata" in result
    assert result["metadata"]["svm_accuracy_original"] >= 0.0
    assert result["metadata"]["svm_accuracy_embeddings"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
