"""
ML services module providing high-level interfaces for model training and prediction.

This module provides service classes that orchestrate the complete machine learning pipeline,
including data loading, preprocessing, EBM training, and SVM classification.

Author: Development Team
Version: 1.0.0
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

from ml_models.ebm_svm import EBMEncoder, EBMTrainer, SVMClassifier, get_embeddings
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Model Management Service
# ============================================================================

class ModelManager:
    """
    Service for managing trained models (persistence and retrieval).
    
    Handles saving and loading trained EBM and SVM models to/from disk
    for later use in predictions.
    
    Attributes:
        models_dir (str): Directory to store trained models
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize ModelManager.
        
        Args:
            models_dir (str): Directory for model storage. Default: 'models'
        """
        self.models_dir = models_dir
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        training_id: str,
        ebm: EBMEncoder,
        svm_original: SVMClassifier,
        svm_embeddings: SVMClassifier,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Save trained models to disk.
        
        Args:
            training_id (str): Unique identifier for this training session
            ebm (EBMEncoder): Trained EBM model
            svm_original (SVMClassifier): SVM trained on original features
            svm_embeddings (SVMClassifier): SVM trained on EBM embeddings
            metadata (Dict): Training metadata (accuracy, params, etc.)
            
        Raises:
            IOError: If model saving fails
        """
        try:
            model_path = Path(self.models_dir) / training_id
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save EBM
            torch.save(ebm.state_dict(), model_path / "ebm_encoder.pth")
            
            # Save SVM models
            with open(model_path / "svm_original.pkl", "wb") as f:
                pickle.dump(svm_original, f)
            
            with open(model_path / "svm_embeddings.pkl", "wb") as f:
                pickle.dump(svm_embeddings, f)
            
            # Save metadata
            with open(model_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Models saved successfully for training_id: {training_id}")
        except IOError as e:
            logger.error(f"Failed to save models: {str(e)}")
            raise
    
    def load_model(self, training_id: str) -> Tuple[EBMEncoder, SVMClassifier, SVMClassifier, Dict]:
        """
        Load trained models from disk.
        
        Args:
            training_id (str): Unique identifier for the training session
            
        Returns:
            Tuple containing:
                - ebm (EBMEncoder): Loaded EBM model
                - svm_original (SVMClassifier): SVM on original features
                - svm_embeddings (SVMClassifier): SVM on embeddings
                - metadata (Dict): Training metadata
                
        Raises:
            FileNotFoundError: If model files don't exist
        """
        try:
            model_path = Path(self.models_dir) / training_id
            
            # Load metadata first to get model config
            with open(model_path / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Reconstruct EBM with saved config
            ebm = EBMEncoder(
                input_dim=metadata["input_dim"],
                hidden_dim=metadata["ebm_config"]["hidden_dim"],
                embedding_dim=metadata["ebm_config"]["embedding_dim"]
            )
            ebm.load_state_dict(torch.load(model_path / "ebm_encoder.pth"))
            ebm.eval()
            
            # Load SVM models
            with open(model_path / "svm_original.pkl", "rb") as f:
                svm_original = pickle.load(f)
            
            with open(model_path / "svm_embeddings.pkl", "rb") as f:
                svm_embeddings = pickle.load(f)
            
            logger.info(f"Models loaded successfully for training_id: {training_id}")
            return ebm, svm_original, svm_embeddings, metadata
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {str(e)}")
            raise


# ============================================================================
# Data Processing Service
# ============================================================================

class DataProcessor:
    """
    Service for data loading and preprocessing.
    
    Handles CSV file loading, feature extraction, data normalization,
    and train/test splitting.
    
    Attributes:
        label_encoders (Dict): Dictionary of label encoders for categorical columns
    """
    
    def __init__(self):
        """Initialize DataProcessor."""
        self.label_encoders = {}
    
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Load CSV file safely.
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid CSV
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {str(e)}")
            raise ValueError(f"Invalid CSV file: {str(e)}")
    
    def extract_features_and_labels(
        self,
        df: pd.DataFrame,
        label_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from dataframe.
        
        Automatically detects label column if not specified.
        Handles categorical features by encoding them.
        
        Args:
            df (pd.DataFrame): Input dataframe
            label_column (Optional[str]): Name of label column. Auto-detected if None
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
            
        Raises:
            ValueError: If no valid label column found
        """
        df = df.copy()
        
        # Auto-detect label column
        if label_column is None:
            possible_labels = ['label', 'target', 'class', 'y', 'species', 'income', 'income_cat']
            for col in possible_labels:
                if col in df.columns:
                    label_column = col
                    break
                    
        if label_column is None:
            label_column = df.columns[-1]
            logger.info(f"Auto-selected last column '{label_column}' as target.")
            
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataframe")
        
        # Extract labels
        # Handle string labels
        if pd.api.types.is_string_dtype(df[label_column]) or pd.api.types.is_object_dtype(df[label_column]):
            df[label_column] = df[label_column].apply(lambda x: x.strip() if isinstance(x, str) else x)
            
        y = df[label_column].values
        
        # Extract features
        X = df.drop(columns=[label_column])
        
        # Handle categorical features
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        return X.values.astype(np.float32), y
    
    def train_test_split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            test_size (float): Test split ratio
            random_state (int): Random seed
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    
    @staticmethod
    def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features using StandardScaler.
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            
        Returns:
            Tuple: (X_train_scaled, X_test_scaled)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled


# ============================================================================
# EBM-SVM Pipeline Service
# ============================================================================

class EBMSVMPipeline:
    """
    Complete pipeline for EBM-SVM integration.
    
    Orchestrates the full workflow: data loading, preprocessing, EBM training,
    and SVM classification on both original and embedded features.
    
    Attributes:
        data_processor (DataProcessor): Data processing service
        model_manager (ModelManager): Model management service
    """
    
    def __init__(self):
        """Initialize the EBM-SVM pipeline."""
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
    
    def train(
        self,
        file_path: str,
        label_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        ebm_config: Optional[Dict] = None,
        svm_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train EBM-SVM pipeline on CSV data.
        
        Args:
            file_path (str): Path to CSV file
            label_column (Optional[str]): Label column name
            test_size (float): Test split ratio
            random_state (int): Random seed
            ebm_config (Optional[Dict]): EBM configuration
            svm_config (Optional[Dict]): SVM configuration
            
        Returns:
            Dict containing training results and model info
            
        Raises:
            ValueError: If data or configuration is invalid
        """
        start_time = datetime.now()
        training_id = start_time.strftime("%Y%m%d_%H%M%S")
        
        try:
            # Default configs
            if ebm_config is None:
                ebm_config = {
                    "hidden_dim": settings.ebm_hidden_dim,
                    "embedding_dim": settings.ebm_embedding_dim,
                    "epochs": settings.ebm_epochs,
                    "learning_rate": settings.ebm_lr,
                    "noise_scale": settings.ebm_noise_scale
                }
            
            if svm_config is None:
                svm_config = {
                    "kernel": settings.svm_kernel,
                    "C": settings.svm_c
                }
            
            # Load and preprocess data
            logger.info("Loading data...")
            df = self.data_processor.load_csv(file_path)
            X, y = self.data_processor.extract_features_and_labels(df, label_column)
            
            input_dim = X.shape[1]
            logger.info(f"Data shape: {X.shape}, Labels: {np.unique(y)}")
            
            # Split data
            X_train, X_test, y_train, y_test = self.data_processor.train_test_split_data(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Normalize
            X_train_scaled, X_test_scaled = self.data_processor.normalize_features(X_train, X_test)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            
            # Train EBM
            logger.info("Training EBM encoder...")
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
            ebm = trainer.train(X_train_tensor, verbose=True)
            
            # Extract embeddings
            logger.info("Extracting embeddings...")
            train_embeddings = get_embeddings(ebm, X_train_tensor)
            test_embeddings = get_embeddings(ebm, X_test_tensor)
            
            # Train SVM on original features
            logger.info("Training SVM on original features...")
            svm_original = SVMClassifier(kernel=svm_config["kernel"], C=svm_config["C"])
            svm_original.fit(X_train_scaled, y_train)
            metrics_original = svm_original.evaluate(X_test_scaled, y_test)
            
# Train SVM on embeddings plus original features (hybrid EBM-SVM approach)
            logger.info("Training SVM on EBM embeddings + original features...")
            
            train_combined = np.hstack((X_train_scaled, train_embeddings))
            test_combined = np.hstack((X_test_scaled, test_embeddings))
            
            svm_embeddings = SVMClassifier(kernel=svm_config["kernel"], C=svm_config["C"])
            svm_embeddings.fit(train_combined, y_train)
            metrics_embeddings = svm_embeddings.evaluate(test_combined, y_test)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare metadata
            metadata = {
                "training_id": training_id,
                "timestamp": start_time.isoformat(),
                "input_dim": input_dim,
                "n_classes": len(np.unique(y)),
                "n_samples": len(X),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "ebm_config": ebm_config,
                "svm_config": svm_config,
                "svm_accuracy_original": float(metrics_original["accuracy"]),
                "svm_accuracy_embeddings": float(metrics_embeddings["accuracy"]),
                "training_time_seconds": training_time
            }
            
            # Save models
            self.model_manager.save_model(
                training_id,
                ebm,
                svm_original,
                svm_embeddings,
                metadata
            )
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return {
                "training_id": training_id,
                "metrics_original": metrics_original,
                "metrics_embeddings": metrics_embeddings,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def predict(
        self,
        training_id: str,
        file_path: str,
        use_embeddings: bool = True
    ) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            training_id (str): ID of trained model
            file_path (str): Path to data file for prediction
            use_embeddings (bool): Use SVM on embeddings (True) or original features (False)
            
        Returns:
            np.ndarray: Predictions
            
        Raises:
            FileNotFoundError: If model or data file not found
        """
        try:
            # Load model
            ebm, svm_original, svm_embeddings, metadata = self.model_manager.load_model(training_id)
            
            # Load and preprocess data
            df = self.data_processor.load_csv(file_path)
            X, _ = self.data_processor.extract_features_and_labels(df)
            
            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Make predictions
            if use_embeddings:
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                embeddings = get_embeddings(ebm, X_tensor)
                combined = np.hstack((X_scaled, embeddings))
                predictions = svm_embeddings.predict(combined)
            else:
                predictions = svm_original.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
