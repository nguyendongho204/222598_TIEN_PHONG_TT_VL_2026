"""
JEPA + SVM Pipeline: JEPA feature extraction + SVM classification

  Bước 1: Train JEPA self-supervised (không cần nhãn)
  Bước 2: Supervised fine-tune JEPA
  Bước 3: Dùng JEPA Encoder để trích embedding
  Bước 4: Train SVM trên embedding
  Bước 5: Đánh giá SVM trên test set
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from jepa_model import JEPA

logger = logging.getLogger(__name__)


class JEPASVMEnsemble:
    """
    JEPA + SVM: Joint Embedding Predictive Architecture + Support Vector Machine

    Pipeline:
      1. MinMaxScaler
      2. Train JEPA (self-supervised, không cần nhãn)
      3. Supervised fine-tune JEPA
      4. Extract embeddings từ JEPA Encoder
      5. Train SVM trên embeddings
    """

    def __init__(self,
                 random_state: int = 42,
                 embedding_dim: int = 32,
                 jepa_epochs: int = 200,
                 jepa_hidden_dims: list = None,
                 device: str = 'cpu'):
        self.random_state = random_state
        self.embedding_dim = embedding_dim
        self.jepa_epochs = jepa_epochs
        self.jepa_hidden_dims = jepa_hidden_dims
        self.device = device

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.jepa = None
        self.svm = None

        self._trained = False
        self._results_cache = None

    def _adjust_epochs(self, n_samples: int) -> int:
        if n_samples < 300:
            return 600
        elif n_samples < 800:
            return 500
        elif n_samples < 2000:
            return 300
        else:
            return self.jepa_epochs

    def run_pipeline(self, X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray,
                     X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        logger.info("\n" + "=" * 70)
        logger.info("JEPA + SVM PIPELINE")
        logger.info("=" * 70)

        n_classes = int(y_train.max()) + 1

        # Scale
        X_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # === JEPA Self-Supervised Training ===
        epochs = self._adjust_epochs(len(X_train))
        logger.info(f"\n[JEPA] Self-supervised training (epochs={epochs}, dim={self.embedding_dim})")

        self.jepa = JEPA(
            input_dim=X_scaled.shape[1],
            embedding_dim=self.embedding_dim,
            hidden_dims=self.jepa_hidden_dims or [64, 32],
            device=self.device
        )

        self.jepa.train(
            X_scaled,
            epochs=epochs,
            batch_size=min(32, len(X_scaled) // 2),
            mask_rate=0.3,
            lr=0.001,
            verbose=True
        )

        # === Supervised Fine-Tune ===
        logger.info(f"\n[JEPA] Supervised fine-tuning")
        self.jepa.train_supervised(
            X_scaled, y_train,
            epochs=max(30, epochs // 5),
            lr=0.0005,
            verbose=True
        )

        # === Extract Embeddings ===
        logger.info(f"\n[JEPA] Extracting embeddings")
        train_emb = self.jepa.extract_features(X_scaled)
        test_emb = self.jepa.extract_features(X_test_scaled)
        logger.info(f"  Train embeddings: {train_emb.shape}")
        logger.info(f"  Test embeddings: {test_emb.shape}")

        # === SVM on JEPA Embeddings ===
        logger.info(f"\n[SVM] Grid search on JEPA embeddings")
        
        svm_base = SVC(kernel='rbf', random_state=self.random_state, probability=True)
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.01, 0.1],
        }
        grid = GridSearchCV(svm_base, param_grid, cv=min(3, len(np.unique(y_train))),
                            scoring='accuracy', n_jobs=1)
        grid.fit(train_emb, y_train)
        self.svm = grid.best_estimator_
        logger.info(f"  Best params: {grid.best_params_}")
        logger.info(f"  Best CV score: {grid.best_score_*100:.2f}%")

        y_pred = self.svm.predict(test_emb)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"  Test accuracy: {accuracy*100:.2f}%")

        # Metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm_arr = np.array(cm)
        per_class_acc = (cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1)).tolist()

        results = {
            'accuracy': accuracy,
            'predictions': y_pred.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm,
            'classification_report': report,
            'per_class_accuracy': per_class_acc,
            'embedding_dim': self.embedding_dim,
        }

        self._results_cache = results
        self._trained = True
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise ValueError("Model not trained yet.")
        X_scaled = self.scaler.transform(X)
        emb = self.jepa.extract_features(X_scaled)
        return self.svm.predict(emb)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise ValueError("Model not trained yet.")
        X_scaled = self.scaler.transform(X)
        emb = self.jepa.extract_features(X_scaled)
        return self.svm.predict_proba(emb)

    def is_trained(self) -> bool:
        return self._trained

    def get_cached_results(self) -> Optional[Dict]:
        return self._results_cache
