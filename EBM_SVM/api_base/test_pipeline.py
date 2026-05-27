import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Quick test
df = pd.read_csv('../data/adult.csv')
print(f"Original dataset: {df.shape}")

# Auto-detect target
target_col = 'income'
y = df[target_col].values
X = df.drop(columns=[target_col]).values

print(f"Before preprocessing: X={X.shape}, y={y.shape}")
print(f"X dtype: {X.dtype}, has strings: {any(isinstance(v, str) for row in X for v in row)}")

# Handle missing values and categorical features
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
print(f"After cleaning: X shape={X.shape}, dtype={X.dtype}")

# Remove NaN rows
valid_mask = ~np.isnan(X).any(axis=1)
X = X[valid_mask]
y = y[valid_mask]
print(f"After removing NaN: X shape={X.shape}, y shape={y.shape}")

# Encode labels
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Encoded labels: {np.unique(y)}")

# Now test pipeline
from ml_models.universal_pipeline import UniversalEBMSVMPipeline

try:
    pipeline = UniversalEBMSVMPipeline(device='cpu')
    print("✓ Pipeline initialized")
    
    X_train, X_test, y_train, y_test = pipeline.load_and_prepare_data(X, y, test_size=0.2)
    print(f"✓ Data split: train {X_train.shape}, test {X_test.shape}")
    
    pipeline.analyze_and_suggest(X_train, y_train)
    print("✓ Analysis done")
    
    baseline = pipeline.train_baseline_svm(X_train, X_test, y_train, y_test)
    print(f"✓ Baseline SVM: {baseline*100:.2f}%")
    
    print("\n✓ Full pipeline working!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

