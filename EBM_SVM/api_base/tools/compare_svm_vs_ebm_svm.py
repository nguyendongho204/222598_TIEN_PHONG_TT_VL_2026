"""
Comparison Demo: Standard SVM vs Optimized EBM-SVM

This script demonstrates the performance improvement of optimized EBM-SVM
over standard SVM on various datasets.

Usage:
    python compare_svm_vs_ebm_svm.py --data-path <path_to_data>
    
Example:
    python compare_svm_vs_ebm_svm.py --data-path data/adult.csv
    python compare_svm_vs_ebm_svm.py --data-path data/breast-cancer.csv

Author: Development Team
Version: 1.0.0
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_models.services.pipeline import EBMSVMPipeline
from ml_models.services.optimized_pipeline import OptimizedEBMSVMPipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Standard SVM Baseline
# ============================================================================

class StandardSVMBaseline:
    """Train standard SVM with default parameters for baseline comparison."""
    
    @staticmethod
    def load_prepare_data(file_path, test_size=0.2, random_state=42):
        """Load and prepare data."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data: {df.shape}")
        
        # Auto-detect label
        label_col = None
        for col in ['label', 'target', 'class', 'y', 'species', 'income']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("Cannot find label column")
        
        # Extract X, y
        y = df[label_col].values
        X = df.drop(columns=[label_col])
        
        # Encode categorical
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        X = X.values.astype(np.float32)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_and_evaluate(X_train, X_test, y_train, y_test):
        """Train standard SVM with default parameters."""
        logger.info("\nTraining Standard SVM (Default Parameters)...")
        logger.info("  kernel: rbf")
        logger.info("  C: 1.0")
        logger.info("  gamma: scale")
        
        start = time.time()
        
        # Train with default parameters
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        svm.fit(X_train, y_train)
        
        # Predict
        y_pred = svm.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        elapsed = time.time() - start
        
        logger.info(f"  Training time: {elapsed:.2f}s")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return {
            "model": "Standard SVM",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "training_time": elapsed
        }


# ============================================================================
# Comparison Function
# ============================================================================

def compare_models(file_path):
    """Compare standard SVM vs optimized EBM-SVM."""
    
    logger.info("\n" + "="*80)
    logger.info("SVM vs OPTIMIZED EBM-SVM COMPARISON")
    logger.info("="*80)
    logger.info(f"Dataset: {file_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # 1. Standard SVM Baseline
        logger.info("\n" + "-"*80)
        logger.info("1. STANDARD SVM BASELINE")
        logger.info("-"*80)
        
        X_train, X_test, y_train, y_test = StandardSVMBaseline.load_prepare_data(file_path)
        results_std = StandardSVMBaseline.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # 2. Optimized EBM-SVM
        logger.info("\n" + "-"*80)
        logger.info("2. OPTIMIZED EBM-SVM")
        logger.info("-"*80)
        logger.info("Configuration:")
        logger.info("  - EBM epochs: 2000 (vs standard)")
        logger.info("  - EBM learning rate: 0.0001")
        logger.info("  - EBM embedding dim: 128")
        logger.info("  - SVM grid search: Enabled")
        logger.info("  - Feature combination: Original + Embeddings")
        logger.info("  - Ensemble: Enabled")
        
        start = time.time()
        
        pipeline = OptimizedEBMSVMPipeline(
            use_tuning=True,
            ensemble_mode=True
        )
        
        results_ebm = pipeline.train(
            file_path,
            feature_selection="all"
        )
        
        elapsed = time.time() - start
        
        results_ebm_summary = {
            "model": "Optimized EBM-SVM",
            "accuracy": results_ebm["metadata"]["best_accuracy"],
            "precision": results_ebm["metrics"]["enhanced"].get("precision", 0),
            "recall": results_ebm["metrics"]["enhanced"].get("recall", 0),
            "f1": results_ebm["metrics"]["enhanced"].get("f1", 0),
            "training_time": elapsed,
            "improvement": results_ebm["metadata"]["improvement"],
            "improvement_pct": results_ebm["metadata"]["improvement_pct"]
        }
        
        # 3. Results Comparison
        logger.info("\n" + "="*80)
        logger.info("COMPARISON RESULTS")
        logger.info("="*80)
        
        logger.info(f"\n{'Metric':<20} {'Standard SVM':<20} {'EBM-SVM':<20} {'Difference':<15}")
        logger.info("-" * 75)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            std_val = results_std[metric]
            ebm_val = results_ebm_summary[metric]
            diff = ebm_val - std_val
            
            logger.info(f"{metric.capitalize():<20} {std_val:<20.4f} {ebm_val:<20.4f} {diff:+.4f}")
        
        logger.info(f"{'Training Time':<20} {results_std['training_time']:<20.2f}s {results_ebm_summary['training_time']:<20.2f}s {results_ebm_summary['training_time']-results_std['training_time']:+.2f}s")
        
        # 4. Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        
        improvement = results_ebm_summary['accuracy'] - results_std['accuracy']
        improvement_pct = (improvement / results_std['accuracy']) * 100 if results_std['accuracy'] > 0 else 0
        
        logger.info(f"\nStandard SVM Accuracy:     {results_std['accuracy']:.4f} ({results_std['accuracy']*100:.2f}%)")
        logger.info(f"Optimized EBM-SVM Accuracy: {results_ebm_summary['accuracy']:.4f} ({results_ebm_summary['accuracy']*100:.2f}%)")
        logger.info(f"\nImprovement:               +{improvement:.4f} (+{improvement_pct:.2f}%)")
        
        if improvement > 0:
            logger.info(f"\n✓ SUCCESS: EBM-SVM outperforms Standard SVM by {improvement_pct:.2f}%")
        else:
            logger.info(f"\n✗ EBM-SVM did not improve over Standard SVM")
        
        logger.info(f"\nTraining time comparison:")
        logger.info(f"  Standard SVM:     {results_std['training_time']:.2f}s")
        logger.info(f"  EBM-SVM:          {results_ebm_summary['training_time']:.2f}s")
        logger.info(f"  Ratio:            {results_ebm_summary['training_time']/results_std['training_time']:.1f}x")
        
        # 5. Save results
        results_file = f"results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            f.write("SVM vs OPTIMIZED EBM-SVM COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Dataset: {file_path}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("RESULTS:\n")
            f.write(f"{'Metric':<20} {'Standard SVM':<20} {'EBM-SVM':<20} {'Difference':<15}\n")
            f.write("-" * 75 + "\n")
            
            for metric in metrics:
                std_val = results_std[metric]
                ebm_val = results_ebm_summary[metric]
                diff = ebm_val - std_val
                f.write(f"{metric.capitalize():<20} {std_val:<20.4f} {ebm_val:<20.4f} {diff:+.4f}\n")
            
            f.write("\n\nSUMMARY:\n")
            f.write(f"Standard SVM Accuracy:        {results_std['accuracy']:.4f}\n")
            f.write(f"Optimized EBM-SVM Accuracy:   {results_ebm_summary['accuracy']:.4f}\n")
            f.write(f"Improvement:                  +{improvement:.4f} (+{improvement_pct:.2f}%)\n")
        
        logger.info(f"\nResults saved to: {results_file}")
        
        return results_std, results_ebm_summary
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# Main Script
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Standard SVM vs Optimized EBM-SVM"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to training data (CSV file)"
    )
    parser.add_argument(
        "--output",
        default="results/comparison.txt",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    try:
        results_std, results_ebm = compare_models(args.data_path)
        return 0 if results_std is not None else 1
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
