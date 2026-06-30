"""Train Adult dataset only and append to ketqua.txt"""
import sys, warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

api_dir = str(Path(__file__).parent.parent / 'api_base')
ml_dir = str(Path(__file__).parent.parent / 'api_base' / 'ml_models')
for p in [api_dir, ml_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_models.jepa_svm_pipeline import JEPASVMEnsemble

SEED = 42
DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

path = DATA_DIR / 'adult.csv'
df = pd.read_csv(path)
print(f'Adult: {df.shape}')

# Auto-detect target: last column
target_col = df.columns[-1]
print(f'Target column: {target_col}')

y = LabelEncoder().fit_transform(df[target_col].astype(str).str.strip())
X_df = df.drop(columns=[target_col])

for c in X_df.select_dtypes(include=['object']).columns:
    X_df[c] = LabelEncoder().fit_transform(X_df[c].astype(str))

X = X_df.fillna(X_df.median(numeric_only=True)).fillna(0).values.astype(np.float32)
y = y.astype(np.int64)

n, nf, nc = len(X), X.shape[1], len(np.unique(y))
print(f'Samples={n} Features={nf} Classes={nc}')

import time
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

t0 = time.time()
emb_dim = max(8, min(nf * 2, 32))
pipe = JEPASVMEnsemble(random_state=SEED, embedding_dim=emb_dim)
res = pipe.run_pipeline(X_tr, X_te, y_tr, y_te)
elapsed = time.time() - t0

acc = res['accuracy'] * 100
print(f'Accuracy: {acc:.2f}% [{elapsed:.0f}s]')

# Append to ketqua.txt
txt_path = RESULTS_DIR / 'ketqua.txt'
lines = txt_path.read_text(encoding='utf-8').split('\n')
# Find the separator line before TRUNG BINH
insert_idx = None
for i, line in enumerate(lines):
    if line.startswith('---'):
        insert_idx = i
        break
if insert_idx:
    new_row = f'{Adult:<22} {n:<7} {nf:<5} {nc:<4} {acc:<8.2f}%{"":>7} {elapsed:<6.0f}s'
    lines.insert(insert_idx, new_row)
    txt_path.write_text('\n'.join(lines), encoding='utf-8')

# Update JSON
json_path = RESULTS_DIR / 'ketqua.json'
with open(json_path, encoding='utf-8') as f:
    data = json.load(f)
data['results'].append({
    'dataset': 'Adult',
    'samples': n,
    'features': nf,
    'classes': nc,
    'accuracy': round(acc, 2),
    'time_seconds': round(elapsed, 1),
})
avg = np.mean([r['accuracy'] for r in data['results']])
print(f'Updated average: {avg:.2f}% (n={len(data["results"])})')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f'Done. Results appended to {txt_path}')
