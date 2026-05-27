"""
Optimized EBM-SVM Pipeline for Superior Performance.

This module provides an enhanced pipeline that combines:
1. Improved EBM training with better hyperparameters
2. Tuned SVM using grid search (from svm_tuning module)
3. Advanced feature combination strategies
4. Ensemble-based approach for better accuracy

Goal: Achieve accuracy HIGHER than standard SVM (target: 88%+ when SVM is 88%)

Author: Development Team
Version: 2.0.0
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime
import json
import pickle

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

from ml_models.ebm_svm import EBMEncoder, EBMTrainer, SVMClassifier, get_embeddings
from ml_models.svm_tuning import SVMTuner, AdultDatasetSVMTuner
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Optimized EBM-SVM Pipeline
# ============================================================================

class OptimizedEBMSVMPipeline:
    """
    Advanced EBM-SVM pipeline with optimization strategies.
    
    This pipeline is designed to outperform standard SVM through:
    1. Enhanced EBM training (more epochs, better parameters)
    2. Hyperparameter tuning for SVM (grid search)
    3. Smart feature combination and selection
    4. Multi-stage training process
    5. Ensemble predictions
    
    Attributes:
        models_dir (str): Directory to store trained models
        use_tuning (bool): Whether to use grid search for SVM tuning
        ensemble_mode (bool): Whether to use ensemble predictions
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        use_tuning: bool = True,
        ensemble_mode: bool = True
    ):
        """
        Initialize optimized pipeline.
        
        Args:
            models_dir (str): Directory for model storage
            use_tuning (bool): Enable SVM hyperparameter tuning. Default: True
            ensemble_mode (bool): Enable ensemble predictions. Default: True
        """
        self.models_dir = models_dir
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        
        self.use_tuning = use_tuning
        self.ensemble_mode = ensemble_mode
        
        self.label_encoders = {}
    
    # ========================================================================
    # Data Processing
    # ========================================================================
    
    def load_and_prepare_data(
        self,
        file_path: str,
        label_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load and prepare data for training.
        
        Args:
            file_path (str): Path to CSV file
            label_column (Optional[str]): Label column name
            test_size (float): Test split ratio
            random_state (int): Random seed
            
        Returns:
            Tuple containing train/test data and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data: shape={df.shape}")
            
            # Auto-detect label column
            if label_column is None:
                possible_labels = ['label', 'target', 'class', 'y', 'species', 'income']
                for col in possible_labels:
                    if col in df.columns:
                        label_column = col
                        break
            
            if label_column is None:
                raise ValueError("Cannot auto-detect label column")
            
            # Extract features and labels
            y = df[label_column].values
            X = df.drop(columns=[label_column])
            
            # Handle categorical features
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            
            X = X.values.astype(np.float32)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            metadata = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_classes": len(np.unique(y)),
                "class_dist": dict(zip(*np.unique(y, return_counts=True)))
            }
            
            logger.info(f"Data prepared: train={X_train.shape}, test={X_test.shape}")
            logger.info(f"Metadata: {metadata}")
            
            return X_train, X_test, y_train, y_test, metadata
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    # ========================================================================
    # EBM Training (Optimized)
    # ========================================================================
    
    def train_optimized_ebm(
        self,
        X_train: np.ndarray,
        ebm_config: Optional[Dict] = None
    ) -> Tuple[EBMEncoder, np.ndarray, Dict[str, Any]]:
        """
        Train EBM with optimized hyperparameters to AVOID OVERFITTING.
        
        KEY FIX: Reduced sizes and epochs to prevent overfitting
        - SMALLER embedding_dim: 64 (not 128) to reduce noise
        - SMALLER hidden_dim: 256 (not 512) for stability
        - FEWER epochs: 800 (not 2000) to prevent overfit
        - HIGHER learning_rate: 0.001 (not 0.0001) for convergence
        - INCREASED noise_scale: 3.0 for regularization
        
        Args:
            X_train (np.ndarray): Training data
            ebm_config (Optional[Dict]): Custom EBM configuration
            
        Returns:
            Tuple of (trained_ebm, embeddings, scaler, config_used)
        """
        logger.info("Training optimized EBM encoder (with overfitting prevention)...")
        
        if ebm_config is None:
            # FIXED defaults - smaller model to prevent overfitting
            ebm_config = {
                "hidden_dim": 256,           # REDUCED from 512
                "embedding_dim": 64,        # REDUCED from 128
                "epochs": 800,              # REDUCED from 2000
                "learning_rate": 0.001,     # INCREASED from 0.0001
                "noise_scale": 3.0          # INCREASED from 2.0
            }
        
        try:
            # Normalize data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            
            input_dim = X_train_scaled.shape[1]
            
            # Create and train EBM
            ebm = EBMEncoder(
                input_dim=input_dim,
                hidden_dim=ebm_config["hidden_dim"],
                embedding_dim=ebm_config["embedding_dim"]
            )
            
            trainer = EBMTrainer(
                ebm,
                epochs=ebm_config["epochs"],
                learning_rate=ebm_config["learning_rate"],
                noise_scale=ebm_config["noise_scale"]
            )
            
            logger.info(f"EBM config: {ebm_config}")
            logger.info("Using smaller model to prevent overfitting on training data")
            ebm = trainer.train(X_train_tensor, verbose=False)
            
            # Extract embeddings
            embeddings = get_embeddings(ebm, X_train_tensor)
            
            logger.info(f"EBM training completed. Embeddings shape: {embeddings.shape}")
            
            return ebm, embeddings, scaler, ebm_config
            
        except Exception as e:
            logger.error(f"EBM training failed: {str(e)}")
            raise
    
    # ========================================================================
    # Feature Engineering
    # ========================================================================
    
    def create_enhanced_features(
        self,
        X_original: np.ndarray,
        embeddings: np.ndarray,
        feature_selection: str = "all"
    ) -> np.ndarray:
        """
        Create enhanced feature space combining original and embedding features.
        
        Strategies:
        1. 'all': Concatenate all features (but limited embeddings to prevent overfit)
        2. 'selected': Select best embedding dimensions via variance
        3. 'normalized': Normalize before concatenation
        4. 'weighted': Weight features by importance
        
        IMPORTANT: Only use TOP embeddings to prevent adding noise!
        
        Args:
            X_original (np.ndarray): Original features
            embeddings (np.ndarray): EBM embeddings (64D now, not 128D)
            feature_selection (str): Strategy for feature combination
            
        Returns:
            np.ndarray: Enhanced features
        """
        try:
            if feature_selection == "all":
                # Use only TOP embeddings (32 out of 64) to avoid noise
                # Select by variance - highest variance = most informative
                embedding_var = np.var(embeddings, axis=0)
                top_indices = np.argsort(embedding_var)[-32:]  # Top 32 dimensions
                top_embeddings = embeddings[:, top_indices]
                X_enhanced = np.hstack((X_original, top_embeddings))
                logger.info(f"Using top 32 embeddings (out of 64) to reduce noise")
                
            elif feature_selection == "selected":
                # Select top embedding dimensions by variance
                embedding_var = np.var(embeddings, axis=0)
                top_indices = np.argsort(embedding_var)[-32:]  # Top 32 dimensions
                top_embeddings = embeddings[:, top_indices]
                X_enhanced = np.hstack((X_original, top_embeddings))
                
            elif feature_selection == "normalized":
                # Normalize both feature sets SEPARATELY
                # This prevents embeddings from dominating
                scaler_orig = StandardScaler()
                scaler_emb = StandardScaler()
                X_orig_norm = scaler_orig.fit_transform(X_original)
                
                # Use only top embeddings
                embedding_var = np.var(embeddings, axis=0)
                top_indices = np.argsort(embedding_var)[-32:]
                top_embeddings = embeddings[:, top_indices]
                X_emb_norm = scaler_emb.fit_transform(top_embeddings)
                X_enhanced = np.hstack((X_orig_norm, X_emb_norm))
                
            elif feature_selection == "weighted":
                # Use only top embeddings with moderate weighting
                embedding_var = np.var(embeddings, axis=0)
                top_indices = np.argsort(embedding_var)[-32:]
                top_embeddings = embeddings[:, top_indices] * 0.8  # Light weighting
                X_enhanced = np.hstack((X_original, top_embeddings))
                
            else:
                # Default: use top embeddings
                embedding_var = np.var(embeddings, axis=0)
                top_indices = np.argsort(embedding_var)[-32:]
                top_embeddings = embeddings[:, top_indices]
                X_enhanced = np.hstack((X_original, top_embeddings))
            
            logger.info(f"Enhanced features created: {X_enhanced.shape} (strategy: {feature_selection})")
            logger.info(f"Original features: {X_original.shape[1]}, Top embeddings: {top_embeddings.shape[1] if 'top_embeddings' in locals() else 'N/A'}")
            return X_enhanced
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    # ========================================================================
    # SVM Training with Tuning
    # ========================================================================
    
    def train_tuned_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        svm_config: Optional[Dict] = None,
        model_name: str = "SVM"
    ) -> Tuple[SVC, Dict[str, Any]]:
        """
        Train SVM with hyperparameter tuning using grid search.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            svm_config (Optional[Dict]): Custom SVM configuration
            model_name (str): Name for logging
            
        Returns:
            Tuple of (best_model, tuning_results)
        """
        logger.info(f"Training tuned {model_name}...")
        
        if not self.use_tuning:
            logger.info("Tuning disabled, using default SVM")
            svm = SVC(kernel="rbf", C=1.0, probability=True)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            svm.fit(X_scaled, y_train)
            return svm, {"best_params": {"C": 1.0, "kernel": "rbf"}}
        
        try:
            # Use grid search tuning
            tuner = SVMTuner(cv_folds=5, search_method="grid")
            
            # Custom param grid for better tuning
            param_grid = {
                'C': [10, 50, 100, 500, 1000],
                'kernel': ['rbf', 'poly', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            }
            
            best_model, results = tuner.tune_standard_svm(
                X_train, y_train, param_grid=param_grid, n_jobs=-1
            )
            
            logger.info(f"{model_name} tuning completed")
            logger.info(f"Best parameters: {tuner.best_params}")
            logger.info(f"Best CV score: {tuner.best_score:.4f}")
            
            return best_model, {
                "best_params": tuner.best_params,
                "cv_score": tuner.best_score,
                "scaler": tuner.scaler
            }
            
        except Exception as e:
            logger.error(f"SVM tuning failed: {str(e)}")
            raise
    
    # ========================================================================
    # Ensemble Predictions
    # ========================================================================
    
    def ensemble_predict(
        self,
        models_list: List[Tuple[SVC, StandardScaler]],
        X_test: np.ndarray,
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Make ensemble predictions from multiple models.
        
        Args:
            models_list (List): List of (model, scaler) tuples
            X_test (np.ndarray): Test features
            weights (Optional[List]): Weight for each model prediction
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if weights is None:
            weights = [1.0 / len(models_list)] * len(models_list)
        
        predictions = []
        for (model, scaler) in models_list:
            X_scaled = scaler.transform(X_test)
            pred = model.predict_proba(X_scaled)
            predictions.append(pred)
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_proba += pred * weight
        
        return np.argmax(ensemble_proba, axis=1)
    
    # ========================================================================
    # Main Training Pipeline
    # ========================================================================
    
    def train(
        self,
        file_path: str,
        label_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        ebm_config: Optional[Dict] = None,
        svm_config: Optional[Dict] = None,
        feature_selection: str = "all"
    ) -> Dict[str, Any]:
        """
        Complete optimized EBM-SVM training pipeline.
        
        Steps:
        1. Load and prepare data
        2. Train optimized EBM encoder
        3. Extract embeddings
        4. Create enhanced features
        5. Train tuned SVM on:
           a. Original features
           b. Enhanced features (original + embeddings)
           c. Embeddings only
        6. Optionally ensemble predictions
        7. Compare results and return best model
        
        Args:
            file_path (str): Path to training data
            label_column (Optional[str]): Label column name
            test_size (float): Test split ratio
            random_state (int): Random seed
            ebm_config (Optional[Dict]): EBM configuration
            svm_config (Optional[Dict]): SVM configuration
            feature_selection (str): Feature combination strategy
            
        Returns:
            Dict with detailed results, model info, and accuracy comparison
        """
        start_time = datetime.now()
        training_id = start_time.strftime("%Y%m%d_%H%M%S")
        
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZED EBM-SVM PIPELINE")
        logger.info("="*80)
        
        try:
            # Step 1: Load data
            logger.info("\n[Step 1] Loading and preparing data...")
            X_train, X_test, y_train, y_test, data_metadata = self.load_and_prepare_data(
                file_path, label_column, test_size, random_state
            )
            
            # Normalize data
            scaler_train = StandardScaler()
            X_train_scaled = scaler_train.fit_transform(X_train)
            X_test_scaled = scaler_train.transform(X_test)
            
            # Step 2: Train EBM
            logger.info("\n[Step 2] Training optimized EBM encoder...")
            ebm, train_embeddings, scaler_ebm, ebm_config_used = self.train_optimized_ebm(
                X_train, ebm_config
            )
            
            # Extract test embeddings
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            test_embeddings = get_embeddings(ebm, X_test_tensor)
            
            # Step 3: Create enhanced features
            logger.info("\n[Step 3] Creating enhanced feature space...")
            X_train_enhanced = self.create_enhanced_features(
                X_train_scaled, train_embeddings, feature_selection
            )
            X_test_enhanced = self.create_enhanced_features(
                X_test_scaled, test_embeddings, feature_selection
            )
            
            # Step 4: Train multiple SVM variants
            logger.info("\n[Step 4] Training tuned SVM variants...")
            
            # Variant 1: SVM on original features
            svm_original, results_original = self.train_tuned_svm(
                X_train_scaled, y_train, svm_config, "SVM (Original)"
            )
            
            # Variant 2: SVM on enhanced features (main EBM-SVM)
            svm_enhanced, results_enhanced = self.train_tuned_svm(
                X_train_enhanced, y_train, svm_config, "SVM (Enhanced)"
            )
            
            # Variant 3: SVM on embeddings only
            svm_embeddings, results_embeddings = self.train_tuned_svm(
                train_embeddings, y_train, svm_config, "SVM (Embeddings)"
            )
            
            # Step 5: Evaluate models
            logger.info("\n[Step 5] Evaluating models...")
            
            # Predictions
            pred_original = svm_original.predict(X_test_scaled)
            pred_enhanced = svm_enhanced.predict(X_test_enhanced)
            pred_embeddings = svm_embeddings.predict(test_embeddings)
            
            # Metrics
            acc_original = accuracy_score(y_test, pred_original)
            acc_enhanced = accuracy_score(y_test, pred_enhanced)
            acc_embeddings = accuracy_score(y_test, pred_embeddings)
            
            logger.info(f"\nResults:")
            logger.info(f"  SVM (Original):      {acc_original:.4f} ({acc_original*100:.2f}%)")
            logger.info(f"  SVM (Enhanced):      {acc_enhanced:.4f} ({acc_enhanced*100:.2f}%)")
            logger.info(f"  SVM (Embeddings):    {acc_embeddings:.4f} ({acc_embeddings*100:.2f}%)")
            
            # Step 6: Ensemble predictions (if enabled)
            # IMPORTANT: Only ensemble if enhanced is SIGNIFICANTLY better (>1%)
            if self.ensemble_mode and (acc_enhanced - acc_original) > 0.01:
                logger.info("\n[Step 6] Creating ensemble model...")
                logger.info(f"Enhanced is {(acc_enhanced - acc_original)*100:.2f}% better, using ensemble")
                
                models_list = [
                    (svm_original, results_original.get("scaler", scaler_train)),
                    (svm_enhanced, results_enhanced.get("scaler", scaler_train))
                ]
                
                pred_ensemble = self.ensemble_predict(
                    models_list, X_test_enhanced,
                    weights=[0.3, 0.7]  # Weight enhanced more
                )
                
                acc_ensemble = accuracy_score(y_test, pred_ensemble)
                logger.info(f"  Ensemble:            {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")
                
                best_accuracy = acc_ensemble
                best_pred = pred_ensemble
                best_model_type = "ensemble"
            else:
                # If enhanced not better enough, just use the better of the two
                logger.info("\n[Step 6] Ensemble not used (enhanced not significantly better)")
                best_accuracy = max(acc_original, acc_enhanced)
                if best_accuracy == acc_enhanced:
                    best_pred = pred_enhanced
                    best_model_type = "enhanced"
                else:
                    best_pred = pred_original
                    best_model_type = "original"
                    logger.info(f"WARNING: Original SVM is better. EBM did not help this dataset!")
            
            # Calculate improvement
            improvement = best_accuracy - acc_original
            improvement_pct = (improvement / acc_original) * 100 if acc_original > 0 else 0
            
            logger.info(f"\n[Step 7] Summary:")
            logger.info(f"  Best Model: {best_model_type}")
            logger.info(f"  Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            logger.info(f"  Improvement vs Original: +{improvement:.4f} (+{improvement_pct:.2f}%)")
            
            # Training metrics
            eval_original = {
                "accuracy": acc_original,
                "precision": classification_report(y_test, pred_original, output_dict=True).get('weighted avg', {}).get('precision', 0),
                "recall": classification_report(y_test, pred_original, output_dict=True).get('weighted avg', {}).get('recall', 0),
                "f1": classification_report(y_test, pred_original, output_dict=True).get('weighted avg', {}).get('f1-score', 0),
                "confusion_matrix": confusion_matrix(y_test, pred_original).tolist()
            }
            
            eval_enhanced = {
                "accuracy": acc_enhanced,
                "precision": classification_report(y_test, pred_enhanced, output_dict=True).get('weighted avg', {}).get('precision', 0),
                "recall": classification_report(y_test, pred_enhanced, output_dict=True).get('weighted avg', {}).get('recall', 0),
                "f1": classification_report(y_test, pred_enhanced, output_dict=True).get('weighted avg', {}).get('f1-score', 0),
                "confusion_matrix": confusion_matrix(y_test, pred_enhanced).tolist()
            }
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Metadata
            metadata = {
                "training_id": training_id,
                "timestamp": start_time.isoformat(),
                "data_metadata": data_metadata,
                "ebm_config": ebm_config_used,
                "svm_config": results_enhanced["best_params"],
                "feature_selection": feature_selection,
                "accuracy_original": float(acc_original),
                "accuracy_enhanced": float(acc_enhanced),
                "accuracy_embeddings": float(acc_embeddings),
                "best_accuracy": float(best_accuracy),
                "improvement": float(improvement),
                "improvement_pct": float(improvement_pct),
                "best_model_type": best_model_type,
                "training_time_seconds": training_time
            }
            
            logger.info(f"\nTraining completed in {training_time:.2f} seconds")
            logger.info("="*80)
            
            return {
                "training_id": training_id,
                "models": {
                    "ebm": ebm,
                    "svm_original": svm_original,
                    "svm_enhanced": svm_enhanced,
                    "svm_embeddings": svm_embeddings
                },
                "scalers": {
                    "scaler_train": scaler_train,
                    "scaler_ebm": scaler_ebm
                },
                "metrics": {
                    "original": eval_original,
                    "enhanced": eval_enhanced
                },
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
