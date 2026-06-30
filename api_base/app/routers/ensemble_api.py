"""
API Router for JEPA + SVM Ensemble Model
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import io
import logging

from ml_models.jepa_svm_pipeline import JEPASVMEnsemble

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ensemble", tags=["ensemble"])

# Global model instance (JEPA+SVM)
ensemble_model: Optional[JEPASVMEnsemble] = None
model_loaded = False


# ═══════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    features: List[List[float]]
    description: Optional[str] = None


class PredictionResponse(BaseModel):
    predictions: List[int]
    confidences: List[List[float]]
    model_info: Dict[str, Any]


class TrainRequest(BaseModel):
    epochs: int = 200


class TrainResponse(BaseModel):
    status: str
    message: str
    training_info: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    status: str
    loaded: bool
    model_type: str
    info: Dict[str, Any]


# ═══════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════

def load_or_create_model():
    global ensemble_model, model_loaded

    if not model_loaded:
        logger.info("[Ensemble API] Loading JEPA + SVM Pipeline")
        ensemble_model = JEPASVMEnsemble(
            random_state=42,
            embedding_dim=32,
            device='cpu'
        )
        logger.info("  Model will train on first request")
        model_loaded = True

    return ensemble_model


def to_native(v):
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: to_native(v2) for k, v2 in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_native(x) for x in v]
    return v


# ═══════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════

@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    logger.info("[Ensemble API] GET /api/ensemble/info")

    try:
        model = load_or_create_model()
        cached = model.get_cached_results()
        cached_native = to_native(cached) if cached else None

        info = {
            'embedding_dim': model.embedding_dim,
            'device': str(model.device),
            'model_trained': model.is_trained(),
            'strategy': 'JEPA feature extraction + SVM classification',
            'cached_results': cached_native,
        }

        return ModelInfoResponse(
            status="success",
            loaded=model_loaded,
            model_type="JEPA + SVM Pipeline",
            info=info
        )

    except Exception as e:
        logger.error(f"[Ensemble API] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    logger.info("[Ensemble API] POST /api/ensemble/train")

    try:
        model = load_or_create_model()

        return TrainResponse(
            status="pending",
            message="Training endpoint requires data upload. Use /train-with-data instead.",
            training_info={
                'requested_epochs': request.epochs,
            }
        )

    except Exception as e:
        logger.error(f"[Ensemble API] Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"[Ensemble API] POST /api/ensemble/predict - {len(request.features)} samples")

    try:
        model = load_or_create_model()

        if not model.is_trained():
            raise ValueError("Model not trained yet. Train the model first.")

        X = np.array(request.features, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("Features must be 2D array (samples x features)")

        predictions = model.predict(X)
        confidences = model.predict_proba(X)

        return PredictionResponse(
            predictions=predictions.tolist(),
            confidences=confidences.tolist(),
            model_info={
                'model_type': 'JEPA + SVM',
                'num_samples': len(X),
                'num_features': X.shape[1],
                'description': request.description or 'Batch prediction'
            }
        )

    except ValueError as e:
        logger.error(f"[Ensemble API] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Ensemble API] Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-csv")
async def predict_csv(file: UploadFile = File(..., description="CSV file with features (no target column)")):
    import io, csv
    logger.info(f"[Ensemble API] POST /api/ensemble/predict-csv - {file.filename}")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        logger.info(f"  Loaded {df.shape[0]} samples, {df.shape[1]} features")

        model = load_or_create_model()
        if not model.is_trained():
            raise ValueError("Model not trained yet. Train the model first.")

        X = df.values.astype(np.float32)
        if X.ndim != 2:
            raise ValueError("Features must be 2D array")

        predictions = model.predict(X)
        confidences = model.predict_proba(X)
        pred_labels = [str(p) for p in predictions]

        result_df = df.copy()
        result_df['prediction'] = pred_labels
        result_df['confidence'] = np.round(np.max(confidences, axis=1), 4)

        unique, counts = np.unique(predictions, return_counts=True)
        stats = {f"class_{int(k)}": int(v) for k, v in zip(unique, counts)}

        return {
            'status': 'success',
            'num_samples': len(X),
            'predictions': pred_labels,
            'confidences': confidences.tolist(),
            'class_distribution': stats,
            'csv': result_df.to_csv(index=False),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Ensemble API] Predict CSV error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def ensemble_status():
    model = load_or_create_model()
    return {
        'trained': model.is_trained(),
        'model_type': 'JEPA + SVM Pipeline',
    }


@router.post("/health")
async def health_check():
    logger.info("[Ensemble API] POST /api/ensemble/health")

    try:
        model = load_or_create_model()

        return {
            'status': 'healthy',
            'model_type': 'JEPA + SVM',
            'model_trained': model.is_trained(),
            'timestamp': str(np.datetime64('now'))
        }

    except Exception as e:
        logger.error(f"[Ensemble API] Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/")
async def ensemble_root():
    return {
        'service': 'JEPA + SVM API',
        'version': '1.0',
        'endpoints': {
            'GET /api/ensemble/': 'This message',
            'GET /api/ensemble/info': 'Get model information',
            'POST /api/ensemble/predict': 'Make predictions (JSON)',
            'POST /api/ensemble/predict-csv': 'Make predictions (CSV upload)',
            'GET /api/ensemble/status': 'Check if model is trained',
            'POST /api/ensemble/health': 'Health check',
        },
        'example_predict': {
            'url': '/api/ensemble/predict',
            'method': 'POST',
            'body': {
                'features': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                'description': 'Sample prediction'
            }
        }
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    logger.info("Ensemble API Router initialized")
    logger.info(f"Available endpoints: {len(router.routes)}")
    for route in router.routes:
        logger.info(f"  - {route.path} [{','.join(route.methods)}]")
