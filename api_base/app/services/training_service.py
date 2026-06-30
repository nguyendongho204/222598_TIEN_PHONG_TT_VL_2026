"""
Service layer cho việc huấn luyện và dự đoán mô hình ML.

Module này chứa logic nghiệp vụ cho huấn luyện, đánh giá và
sử dụng các mô hình ML. Nó tách biệt logic khỏi tầng API endpoint.

Author: Development Team
Version: 1.0.0
"""

import logging
import io
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from app.constants import (
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    STRATIFY_THRESHOLD,
    RANDOM_SEED,
    ERROR_DATASET_PROCESSING,
    ERROR_MODEL_TRAINING,
)
from app.schemas import TrainingResponse, ErrorResponse

logger = logging.getLogger(__name__)


class DataPreprocessingService:
    """
    Service cho việc tải và tiền xử lý dữ liệu.
    
    Xử lý tải CSV, mã hóa, chuẩn hóa và chia train/test/val.
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Khởi tạo service tiền xử lý.
        
        Args:
            random_state: Seed ngẫu nhiên để tái lập kết quả
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = None
    
    def load_csv(self, file_content: bytes) -> pd.DataFrame:
        """
        Tải file CSV từ dữ liệu bytes.
        
        Args:
            file_content: Nội dung file CSV dạng bytes
            
        Returns:
            Pandas DataFrame
            
        Raises:
            ValueError: Nếu không thể parse CSV
        """
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            error_msg = f"Failed to load CSV: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def encode_features(self, X: np.ndarray, is_train: bool = True) -> np.ndarray:
        """
        Mã hóa đặc trưng dạng category sang số bằng OneHotEncoder.
        
        Chia dữ liệu TRƯỚC khi mã hóa để tránh rò rỉ dữ liệu:
        - fit encoder trên TRAIN only
        - transform val/test với encoder đã fit
        
        Args:
            X: Mảng đặc trưng
            is_train: Có phải tập train không (fit encoders nếu True)
            
        Returns:
            Mảng đặc trưng đã mã hóa (numeric giữ nguyên, categorical one-hot)
        """
        if is_train:
            self.cat_indices_ = []
            self.num_indices_ = []
            for col_idx in range(X.shape[1]):
                try:
                    X[:, col_idx].astype(float)
                    self.num_indices_.append(col_idx)
                except (ValueError, TypeError):
                    self.cat_indices_.append(col_idx)
        
        parts = []
        if self.num_indices_:
            num_part = np.zeros((X.shape[0], len(self.num_indices_)), dtype=np.float32)
            for i, col_idx in enumerate(self.num_indices_):
                col = X[:, col_idx]
                converted = []
                for val in col:
                    try:
                        converted.append(float(val))
                    except (ValueError, TypeError):
                        converted.append(np.nan)
                arr = np.array(converted, dtype=np.float32)
                nan_mask = np.isnan(arr)
                if nan_mask.any():
                    median_val = np.nanmedian(arr)
                    if np.isnan(median_val):
                        median_val = 0.0
                    arr[nan_mask] = median_val
                num_part[:, i] = arr
            parts.append(num_part)
        if self.cat_indices_:
            X_cat = X[:, self.cat_indices_].astype(str)
            if is_train:
                self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                parts.append(self.ohe_.fit_transform(X_cat))
            else:
                parts.append(self.ohe_.transform(X_cat))
        
        return np.hstack(parts).astype(np.float32)
    
    def encode_target(self, y: np.ndarray, is_train: bool = True) -> np.ndarray:
        """
        Mã hóa nhãn mục tiêu (target).
        
        Args:
            y: Nhãn mục tiêu
            is_train: Có phải tập train không
            
        Returns:
            Nhãn mục tiêu đã mã hóa
        """
        y_str = np.array([str(v).strip() for v in y.astype(str)])
        
        if is_train:
            self.target_encoder = LabelEncoder()
            return self.target_encoder.fit_transform(y_str).astype(np.int64)
        else:
            if self.target_encoder is None:
                raise ValueError("Target encoder not fitted yet")
            return self.target_encoder.transform(y_str).astype(np.int64)
    
    def split_data(self, X: np.ndarray, y: np.ndarray,
                    test_size: float = DEFAULT_TEST_SIZE) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, 
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Chia dữ liệu thành train, validation và test sets.
        
        Sử dụng stratification nếu có thể, fallback sang random split
        nếu không thể áp dụng stratification.
        
        Args:
            X: Đặc trưng
            y: Nhãn
            test_size: Tỷ lệ tập test
            
        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            # Check if stratification is possible
            # Kiểm tra xem có thể dùng stratification không
            min_samples_per_class = min(np.bincount(y))
            use_stratify = min_samples_per_class >= STRATIFY_THRESHOLD
            
            # First split: train vs test
            # Lần chia đầu: train vs test
            stratify_arg = y if use_stratify else None
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_arg
            )
            
            # Second split: train vs val
            # Lần chia thứ hai: train vs val
            if use_stratify:
                stratify_arg = y_temp
            else:
                stratify_arg = None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=DEFAULT_VAL_SIZE,
                random_state=self.random_state,
                stratify=stratify_arg
            )
            
            logger.info(
                f"Data split - Train: {X_train.shape[0]}, "
                f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            error_msg = f"Error splitting data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def normalize(self, X_train: np.ndarray, X_val: np.ndarray, 
                  X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chuẩn hóa đặc trưng bằng StandardScaler.
        
        Args:
            X_train: Đặc trưng huấn luyện
            X_val: Đặc trưng validation
            X_test: Đặc trưng test
            
        Returns:
            Tuple (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("Data normalized successfully")
            return X_train_scaled, X_val_scaled, X_test_scaled
            
        except Exception as e:
            error_msg = f"Error normalizing data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class ModelTrainingService:
    """
    Service cho việc huấn luyện mô hình ML.
    
    Điều phối toàn bộ pipeline huấn luyện bao gồm tiền xử lý,
    khởi tạo mô hình và đánh giá.
    """
    
    def __init__(self, preprocessing_service: DataPreprocessingService):
        """
        Khởi tạo service huấn luyện.
        
        Args:
            preprocessing_service: Instance của DataPreprocessingService
        """
        self.preprocessing = preprocessing_service
        self.ensemble = None
    
    def train_ensemble(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        try:
            from ml_models.jepa_svm_pipeline import JEPASVMEnsemble
            
            logger.info(f"Initializing JEPA+SVM (seed={random_state})...")
            self.ensemble = JEPASVMEnsemble(
                random_state=random_state,
                device="cpu"
            )
            
            logger.info("Starting JEPA+SVM pipeline...")
            results = self.ensemble.run_pipeline(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
            
            logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Error training ensemble: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class TrainingOrchestrationService:
    """
    Service chính điều phối toàn bộ workflow huấn luyện.
    
    Kết hợp các service tiền xử lý và huấn luyện để cung cấp
    một interface duy nhất cho API endpoint.
    """
    
    def __init__(self):
        """Khởi tạo service điều phối."""
        self.preprocessing = DataPreprocessingService()
        self.training = ModelTrainingService(self.preprocessing)
    
    def execute_training(
        self,
        file_content: bytes,
        test_size: float = DEFAULT_TEST_SIZE,
        num_runs: int = 5,
        dataset_name: str = "unknown"
    ) -> Dict[str, Any]:
        try:
            n_folds = max(2, min(num_runs, 10))
            logger.info(f"Training dataset: {dataset_name} ({n_folds}-fold Stratified K-Fold)")

            # Load CSV
            logger.info("Step 1/4: Loading CSV...")
            df = self.preprocessing.load_csv(file_content)

            # Preprocess
            logger.info("Step 2/4: Preprocessing data...")
            target_col = df.columns[-1]
            for candidate in ['label', 'class', 'target', 'class_label', 'diagnosis', 'Label', 'Class', 'Target']:
                if candidate in df.columns:
                    target_col = candidate
                    break
            for candidate in ['label', 'class', 'target', 'class_label', 'diagnosis', 'Label', 'Class', 'Target']:
                if candidate in df.columns and candidate != target_col:
                    if df[candidate].dtype == 'object' or df[candidate].nunique() < 20:
                        target_col = candidate
                        break

            logger.info(f"  Target column: '{target_col}' (unique values: {df[target_col].nunique()})")
            logger.info(f"  Feature columns: {[c for c in df.columns if c != target_col]}")
            X_raw = df.drop(columns=[target_col]).values
            y_raw = df[target_col].values
            y = self.preprocessing.encode_target(y_raw, is_train=True)

            # Encode features on all data
            X_encoded = self.preprocessing.encode_features(X_raw, is_train=True)

            # K-Fold: Stratified nếu có thể, nếu không thì dùng KFold thường
            from sklearn.model_selection import StratifiedKFold, KFold
            logger.info(f"Step 3/4: K-Fold ({n_folds} folds)...")
            min_samples = min(np.bincount(y))
            use_stratified = min_samples >= n_folds
            if not use_stratified:
                logger.warning(f"  Smallest class has {min_samples} samples < {n_folds}. Using KFold (non-stratified).")

            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.preprocessing.random_state) if use_stratified else KFold(n_splits=n_folds, shuffle=True, random_state=self.preprocessing.random_state)

            all_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_encoded, y)):
                logger.info(f"\nFold {fold_idx+1}/{n_folds}")
                logger.info(f"{'='*60}")
                X_train_fold = X_encoded[train_idx]
                X_test_fold = X_encoded[test_idx]
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                logger.info(f"  Train: {X_train_fold.shape[0]} samples, Test: {X_test_fold.shape[0]} samples")

                result = self.training.train_ensemble(
                    X_train=X_train_fold, X_val=X_test_fold, X_test=X_test_fold,
                    y_train=y_train_fold, y_val=y_test_fold, y_test=y_test_fold,
                    random_state=self.preprocessing.random_state + fold_idx
                )
                all_results.append(result)

            # Aggregate results across folds
            train_results = all_results[-1]
            logger.info(f"KEYS: {list(train_results.keys())}")

            acc_vals = [r['accuracy'] for r in all_results]
            accuracy_mean = float(np.mean(acc_vals))
            accuracy_std = float(np.std(acc_vals))
            logger.info(f"\nK-Fold results ({n_folds} folds):")
            logger.info(f"  accuracy: {accuracy_mean*100:.2f}% ± {accuracy_std*100:.2f}%")

            response = {
                "status": "success",
                "dataset_name": dataset_name.replace(".csv", ""),
                "samples": int(X_raw.shape[0]),
                "features": int(X_encoded.shape[1]),
                "classes": len(np.unique(y)),
                "accuracy": accuracy_mean,
                "accuracy_std": accuracy_std,
                "per_fold_accuracies": [float(v) for v in acc_vals],
                "embedding_dim": int(train_results.get("embedding_dim", X_encoded.shape[1])),
                "precision": train_results.get("precision", []),
                "recall": train_results.get("recall", []),
                "f1_score": train_results.get("f1_score", []),
                "support": train_results.get("support", []),
                "confusion_matrix": train_results.get("confusion_matrix", []),
                "classification_report": train_results.get("classification_report", {}),
                "per_class_accuracy": train_results.get("per_class_accuracy", []),
            }

            logger.info(f"Training completed: {response}")
            return response

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during training: {str(e)}")
            raise ValueError(f"Training failed: {str(e)}")
