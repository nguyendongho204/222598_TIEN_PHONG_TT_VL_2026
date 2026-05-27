"""
Adult Dataset SVM Training and Tuning Demo.

This script demonstrates how to:
1. Load Adult income dataset
2. Preprocess and prepare data
3. Tune standard SVM (target: 83-85% accuracy)
4. Tune improved SVM with EBM embeddings (target: 86-88% accuracy)
5. Compare results

Usage:
    python train_adult_svm.py [--data-path PATH] [--output-path PATH]

References:
    - Dataset: UCI ML Repository - Adult (Income) Dataset
    - Standard SVM baseline: ~83-85% accuracy
    - Improved SVM (with feature engineering): ~86-88% accuracy

Author: Development Team
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import urllib.request
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_models.ebm_svm import EBMEncoder, EBMTrainer, SVMClassifier, get_embeddings
from ml_models.svm_tuning import SVMTuner, AdultDatasetSVMTuner
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

class AdultDataLoader:
    """
    Utility for loading and preprocessing Adult income dataset.
    
    Handles:
    - Loading from UCI ML repository or local file
    - Missing value handling
    - Categorical feature encoding
    - Target variable preparation
    """
    
    # Adult dataset URL from UCI ML Repository
    ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    
    # Column names
    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    # Categorical columns
    CATEGORICAL_COLS = [
        'workclass', 'education', 'marital_status', 'occupation',
        'relationship', 'race', 'sex', 'native_country'
    ]
    
    @staticmethod
    def download_dataset(output_path: str = "data/adult.csv") -> str:
        """
        Download Adult dataset from UCI ML Repository.
        
        Args:
            output_path (str): Path to save the dataset
            
        Returns:
            str: Path to saved dataset
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            logger.info(f"Downloading Adult dataset to {output_path}...")
            urllib.request.urlretrieve(AdultDataLoader.ADULT_URL, output_path)
            logger.info(f"Dataset downloaded successfully: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            raise
    
    @staticmethod
    def load_and_preprocess(
        file_path: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load and preprocess Adult dataset.
        
        Args:
            file_path (str): Path to Adult dataset CSV
            test_size (float): Test split ratio
            random_state (int): Random seed
            
        Returns:
            Tuple containing:
                - X_train (np.ndarray): Training features
                - X_test (np.ndarray): Test features
                - y_train (np.ndarray): Training labels
                - y_test (np.ndarray): Test labels
                - metadata (Dict): Dataset information
                
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        logger.info(f"Loading Adult dataset from {file_path}...")
        
        try:
            # Load data
            df = pd.read_csv(file_path, names=AdultDataLoader.COLUMNS, sep=',\s*', engine='python')
            
            logger.info(f"Dataset loaded: shape={df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Remove leading/trailing whitespace
            df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
            
            # Handle missing values (marked as '?')
            df = df.replace('?', np.nan)
            df = df.dropna()
            
            logger.info(f"After handling missing values: {df.shape}")
            
            # Prepare target variable
            y = (df['income'] == '>50K').astype(int).values
            
            # Prepare features
            X = df.drop(columns=['income', 'fnlwgt'])  # Remove target and weight
            
            # Encode categorical features
            label_encoders = {}
            for col in AdultDataLoader.CATEGORICAL_COLS:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le
            
            # Convert to numeric
            X = X.astype(np.float32).values
            
            logger.info(f"Features prepared: shape={X.shape}")
            logger.info(f"Target distribution: {np.bincount(y)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Train/Test split: {X_train.shape} / {X_test.shape}")
            
            metadata = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_classes": len(np.unique(y)),
                "class_distribution": {
                    "<=50K": np.sum(y == 0),
                    ">50K": np.sum(y == 1)
                },
                "feature_columns": df.drop(columns=['income', 'fnlwgt']).columns.tolist()
            }
            
            return X_train, X_test, y_train, y_test, metadata
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise


# ============================================================================
# Training Functions
# ============================================================================

def train_standard_svm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train standard SVM on Adult dataset (target: 83-85%).
    
    Args:
        X_train, X_test, y_train, y_test: Train/test data
        
    Returns:
        Dict: Training results including accuracy and best parameters
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING STANDARD SVM (Target: 83-85% accuracy)")
    logger.info("="*70)
    
    try:
        # Create and tune SVM
        adult_tuner = AdultDatasetSVMTuner(cv_folds=5)
        best_model, tuning_results = adult_tuner.tune_adult_standard_svm(X_train, y_train)
        
        # Evaluate on test set
        eval_results = adult_tuner.tuner.evaluate_model(
            best_model, X_test, y_test, model_name="Standard SVM"
        )
        
        logger.info(f"\nStandard SVM Test Results:")
        logger.info(f"  - Accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
        logger.info(f"  - Precision: {eval_results['precision']:.4f}")
        logger.info(f"  - Recall: {eval_results['recall']:.4f}")
        logger.info(f"  - F1-Score: {eval_results['f1']:.4f}")
        logger.info(f"\nBest Parameters:")
        for key, value in adult_tuner.tuner.best_params.items():
            logger.info(f"  - {key}: {value}")
        
        return {
            "model_type": "standard_svm",
            "accuracy": eval_results['accuracy'],
            "precision": eval_results['precision'],
            "recall": eval_results['recall'],
            "f1": eval_results['f1'],
            "best_params": adult_tuner.tuner.best_params,
            "cv_score": adult_tuner.tuner.best_score,
            "model": best_model,
            "scaler": adult_tuner.tuner.scaler,
            "eval_results": eval_results
        }
        
    except Exception as e:
        logger.error(f"Standard SVM training failed: {str(e)}")
        raise


def train_improved_svm_with_ebm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train improved SVM with EBM embeddings (target: 86-88%).
    
    Args:
        X_train, X_test, y_train, y_test: Train/test data
        
    Returns:
        Dict: Training results including accuracy and best parameters
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING IMPROVED SVM WITH EBM EMBEDDINGS (Target: 86-88%)")
    logger.info("="*70)
    
    try:
        # Normalize features for EBM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train EBM encoder
        logger.info("Training EBM encoder for feature extraction...")
        input_dim = X_train_scaled.shape[1]
        ebm = EBMEncoder(input_dim=input_dim, hidden_dim=512, embedding_dim=128)
        
        trainer = EBMTrainer(
            ebm,
            epochs=1000,
            learning_rate=0.0001,
            noise_scale=2.0
        )
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        ebm = trainer.train(X_train_tensor, verbose=False)
        
        logger.info("EBM training completed")
        
        # Extract embeddings
        logger.info("Extracting embeddings...")
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        train_embeddings = get_embeddings(ebm, X_train_tensor)
        test_embeddings = get_embeddings(ebm, X_test_tensor)
        
        logger.info(f"Embeddings shape: {train_embeddings.shape}")
        
        # Tune improved SVM with combined features
        adult_tuner = AdultDatasetSVMTuner(cv_folds=5)
        best_model, tuning_results = adult_tuner.tune_adult_improved_svm(
            X_train_scaled, y_train,
            X_embeddings_train=train_embeddings
        )
        
        # Evaluate on test set with combined features
        X_test_combined = np.hstack((X_test_scaled, test_embeddings))
        eval_results = adult_tuner.tuner.evaluate_model(
            best_model, X_test_combined, y_test, model_name="Improved SVM"
        )
        
        logger.info(f"\nImproved SVM Test Results:")
        logger.info(f"  - Accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
        logger.info(f"  - Precision: {eval_results['precision']:.4f}")
        logger.info(f"  - Recall: {eval_results['recall']:.4f}")
        logger.info(f"  - F1-Score: {eval_results['f1']:.4f}")
        logger.info(f"\nBest Parameters:")
        for key, value in adult_tuner.tuner.best_params.items():
            logger.info(f"  - {key}: {value}")
        
        return {
            "model_type": "improved_svm_ebm",
            "accuracy": eval_results['accuracy'],
            "precision": eval_results['precision'],
            "recall": eval_results['recall'],
            "f1": eval_results['f1'],
            "best_params": adult_tuner.tuner.best_params,
            "cv_score": adult_tuner.tuner.best_score,
            "model": best_model,
            "ebm": ebm,
            "scaler": scaler,
            "eval_results": eval_results
        }
        
    except Exception as e:
        logger.error(f"Improved SVM training failed: {str(e)}")
        raise


def compare_models(standard_results: Dict, improved_results: Dict) -> None:
    """
    Compare standard and improved SVM results.
    
    Args:
        standard_results (Dict): Standard SVM results
        improved_results (Dict): Improved SVM results
    """
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON")
    logger.info("="*70)
    
    logger.info(f"\n{'Metric':<15} {'Standard SVM':<20} {'Improved SVM':<20} {'Improvement':<15}")
    logger.info("-" * 70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        std_val = standard_results.get(metric, 0)
        imp_val = improved_results.get(metric, 0)
        diff = imp_val - std_val
        
        logger.info(f"{metric.capitalize():<15} {std_val:<20.4f} {imp_val:<20.4f} {diff:+.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Standard SVM achieved: {standard_results['accuracy']*100:.2f}% accuracy")
    logger.info(f"Improved SVM achieved: {improved_results['accuracy']*100:.2f}% accuracy")
    logger.info(f"Improvement: +{(improved_results['accuracy'] - standard_results['accuracy'])*100:.2f}%")


# ============================================================================
# Main Script
# ============================================================================

def main():
    """
    Main function to run complete Adult dataset SVM training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Train and tune SVM on Adult income dataset"
    )
    parser.add_argument(
        "--data-path",
        default="data/adult.csv",
        help="Path to Adult dataset CSV file"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from UCI ML Repository"
    )
    parser.add_argument(
        "--output-path",
        default="results/adult_svm_results.txt",
        help="Path to save results"
    )
    
    args = parser.parse_args()
    
    try:
        # Download dataset if requested
        if args.download or not os.path.exists(args.data_path):
            logger.info("Downloading Adult dataset...")
            args.data_path = AdultDataLoader.download_dataset(args.data_path)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing Adult dataset...")
        X_train, X_test, y_train, y_test, metadata = AdultDataLoader.load_and_preprocess(
            args.data_path
        )
        
        logger.info(f"\nDataset Information:")
        for key, value in metadata.items():
            if key != "feature_columns":
                logger.info(f"  {key}: {value}")
        
        # Train models
        standard_results = train_standard_svm(X_train, X_test, y_train, y_test)
        improved_results = train_improved_svm_with_ebm(X_train, X_test, y_train, y_test)
        
        # Compare results
        compare_models(standard_results, improved_results)
        
        # Save results
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            f.write("Adult Dataset SVM Training Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("Dataset Information:\n")
            for key, value in metadata.items():
                if key != "feature_columns":
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n\nStandard SVM Results:\n")
            f.write(f"  Accuracy: {standard_results['accuracy']:.4f} ({standard_results['accuracy']*100:.2f}%)\n")
            f.write(f"  Best Parameters: {standard_results['best_params']}\n")
            
            f.write("\n\nImproved SVM Results:\n")
            f.write(f"  Accuracy: {improved_results['accuracy']:.4f} ({improved_results['accuracy']*100:.2f}%)\n")
            f.write(f"  Best Parameters: {improved_results['best_params']}\n")
            
            f.write("\n\nImprovement:\n")
            f.write(f"  +{(improved_results['accuracy'] - standard_results['accuracy'])*100:.2f}%\n")
        
        logger.info(f"\nResults saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
