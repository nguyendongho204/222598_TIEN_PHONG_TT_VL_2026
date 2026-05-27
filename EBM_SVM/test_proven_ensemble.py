"""
Test ProvenPCASVMEnsemble to ensure it works correctly
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "api_base"))

from api_base.ml_models.proven_ensemble import ProvenPCASVMEnsemble


def test_dataset(name, X, y):
    """Test on a single dataset"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Shape: {X.shape}, Classes: {len(np.unique(y))}")
    print(f"{'='*60}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Run ensemble
    ensemble = ProvenPCASVMEnsemble(random_state=42)
    result = ensemble.run_pipeline(X_train, X_test, y_train, y_test)
    
    # Check results
    baseline = result['baseline_accuracy']
    final = result['final_accuracy']
    improvement = result['improvement_pct']
    approach = result['approach']
    
    print(f"\nResults:")
    print(f"  Baseline SVM: {baseline*100:.2f}%")
    print(f"  Final:       {final*100:.2f}%")
    print(f"  Improvement: +{improvement:.2f}%")
    print(f"  Approach:    {approach}")
    
    # Validate
    assert final >= baseline, f"ERROR: Final accuracy {final} < Baseline {baseline}"
    print(f"✓ PASS: Final >= Baseline")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING PROVEN PCA + SVM ENSEMBLE")
    print("="*60)
    
    # Test 1: Iris
    X, y = load_iris(return_X_y=True)
    test_dataset("Iris", X, y)
    
    # Test 2: Breast Cancer
    X, y = load_breast_cancer(return_X_y=True)
    test_dataset("Breast Cancer", X, y)
    
    print(f"\n{'='*60}")
    print("ALL TESTS PASSED ✓")
    print(f"{'='*60}\n")
