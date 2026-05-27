"""
Test ProvenEnsemble on all datasets from data folder
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).parent / "api_base"))

from api_base.ml_models.proven_ensemble import ProvenPCASVMEnsemble


def load_dataset(csv_path):
    """Load and preprocess dataset"""
    df = pd.read_csv(csv_path)
    
    # Auto-detect target column
    target_col = None
    for col in ['label', 'target', 'class', 'y', 'species', 'income', 'income_cat']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
    
    # Extract
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    
    # Handle strings in X
    X_clean = []
    for row in X:
        row_clean = []
        for val in row:
            if isinstance(val, str):
                val = val.strip()
                if val == '?':
                    val = np.nan
                else:
                    try:
                        val = float(val)
                    except:
                        val = 0
            row_clean.append(val)
        X_clean.append(row_clean)
    
    X = np.array(X_clean, dtype=float)
    
    # Remove NaN rows
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Encode labels
    if y.dtype == object or pd.api.types.is_string_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y


def test_dataset(csv_path, limit_samples=None):
    """Test on a dataset"""
    name = Path(csv_path).stem
    
    try:
        # Load
        X, y = load_dataset(csv_path)
        
        if limit_samples and len(X) > limit_samples:
            print(f"  [Sample] {len(X)} -> {limit_samples} samples")
            X, _, y, _ = train_test_split(X, y, train_size=limit_samples, 
                                          stratify=y, random_state=42)
        
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"Shape: {X.shape}, Classes: {len(np.unique(y))}")
        print(f"{'='*60}")
        
        # Split (with fallback to non-stratified if stratified fails)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # Fallback if stratified split fails (e.g., tiny class size)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
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
        ensemble_acc = result['ensemble_accuracy']
        final = result['final_accuracy']
        improvement = result['improvement_pct']
        approach = result['approach']
        
        print(f"\nResults:")
        print(f"  Baseline SVM: {baseline*100:.2f}%")
        print(f"  Ensemble:     {ensemble_acc*100:.2f}%")
        print(f"  Final:        {final*100:.2f}%")
        print(f"  Improvement:  +{improvement:.2f}%")
        print(f"  Approach:     {approach}")
        
        # Validate
        if final < baseline - 0.0001:  # Allow tiny floating point error
            print(f"❌ FAIL: Final accuracy {final} < Baseline {baseline}")
            return False
        else:
            print(f"✓ PASS: Final >= Baseline")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING PROVEN PCA + SVM ENSEMBLE ON ALL DATASETS")
    print("="*60)
    
    data_folder = Path(__file__).parent / "data"
    datasets = [
        "wine.csv",
        "Iris.csv",
        "breast-cancer.csv",
        "moons.csv",
        "circles.csv",
        "adult.csv"  # Large dataset - will sample
    ]
    
    results = {}
    for dataset in datasets:
        path = data_folder / dataset
        if path.exists():
            # Limit adult.csv sampling to 500
            limit = 500 if "adult" in dataset.lower() else None
            results[dataset] = test_dataset(path, limit_samples=limit)
        else:
            print(f"\n⊘ SKIP: {dataset} not found")
            results[dataset] = None
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if v is not None)
    
    for dataset, result in results.items():
        status = "✓ PASS" if result is True else ("❌ FAIL" if result is False else "⊘ SKIP")
        print(f"{status}: {dataset}")
    
    print(f"\n{'='*60}")
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total}) ✓")
    else:
        print(f"TESTS FAILED ({passed}/{total}) ❌")
    print(f"{'='*60}\n")
