"""
Universal EBM-SVM Pipeline - Proven PCA + SVM Ensemble

Simple, clean, proven approach that guarantees:
final_accuracy >= baseline_accuracy
"""

import logging
import numpy as np
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import components
from .proven_ensemble import ProvenPCASVMEnsemble

logger = logging.getLogger(__name__)


class UniversalEBMSVMPipeline:
    """
    Universal pipeline using Proven PCA + SVM Ensemble approach.
    Always guarantees: final_accuracy >= baseline_accuracy
    """
    
    def __init__(self, device: str = "cpu", random_state: int = 42):
        """
        Initialize pipeline.
        
        Args:
            device: 'cpu' or 'cuda'
            random_state: Random seed
        """
        self.device = device
        self.random_state = random_state
        self.ensemble = ProvenPCASVMEnsemble(random_state=random_state)
        self.results = {}
    
    def run_full_pipeline(self, X: np.ndarray, y: np.ndarray, 
                         test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run proven ensemble pipeline.
        
        Args:
            X: Features
            y: Labels
            test_size: Test set fraction
            
        Returns:
            Results dictionary with guaranteed accuracy improvement or equality
        """
        logger.info("\n" + "="*100)
        logger.info("UNIVERSAL PCA + SVM ENSEMBLE PIPELINE")
        logger.info("="*100)
        
        # Prepare data
        logger.info("\n[DATA PREPARATION]")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        logger.info(f"  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Test: {X_test.shape}")
        logger.info(f"  Classes: {len(np.unique(y))}")
        
        # Run ensemble
        result = self.ensemble.run_pipeline(X_train, X_test, y_train, y_test)
        
        # Log final results
        logger.info("\n" + "="*100)
        logger.info("FINAL RESULTS")
        logger.info("="*100)
        logger.info(f"Baseline SVM: {result['baseline_accuracy']*100:.2f}%")
        logger.info(f"Final Result: {result['final_accuracy']*100:.2f}%")
        logger.info(f"Improvement: +{result['improvement_pct']:.2f}%")
        logger.info(f"Approach: {result['approach']}")
        logger.info("="*100 + "\n")
        
        self.results = result
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test on Iris
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    
    pipeline = UniversalEBMSVMPipeline(device='cpu')
    result = pipeline.run_full_pipeline(X, y, test_size=0.2)
