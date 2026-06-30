"""
Benchmark runner: all UCI datasets
Skips Adult (too slow for JEPA), runs rest.
"""

import sys, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import sys
api_dir = str(Path(__file__).parent.parent / 'api_base')
ml_dir = str(Path(__file__).parent.parent / 'api_base' / 'ml_models')
for p in [api_dir, ml_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_models.jepa_svm_pipeline import JEPASVMEnsemble

SEED = 42
DATA_DIR = Path('D:/Nam4/ThucTap/EBM_SVM/data')


def load_prepare(path):
    df = pd.read_csv(path)
    target = 'label' if 'label' in df.columns else df.columns[-1]
    if 'winequality' in str(path):
        y = (df[target].values >= 6).astype(int)
    else:
        y = LabelEncoder().fit_transform(df[target].astype(str).str.strip())
    drop = [target] + [c for c in df.columns if c.lower() in ['id', 'ids']]
    X = df.drop(columns=drop).copy()
    for c in X.select_dtypes(include=['object']).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.fillna(0).values.astype(np.float32)
    return X, y


datasets = [
    ('abalone.csv', 'Abalone'),
    ('balance.csv', 'Balance'),
    ('banknote.csv', 'Banknote'),
    ('breast-cancer.csv', 'Breast Cancer'),
    ('car.csv', 'Car'),
    ('dermatology.csv', 'Dermatology'),
    ('ecoli.csv', 'Ecoli'),
    ('glass.csv', 'Glass'),
    ('haberman.csv', 'Haberman'),
    ('heart.csv', 'Heart'),
    ('ionosphere.csv', 'Ionosphere'),
    ('iris.csv', 'Iris'),
    ('liver.csv', 'Liver'),
    ('mushroom.csv', 'Mushroom'),
    ('optical.csv', 'Optical'),
    ('page-blocks.csv', 'Page Blocks'),
    ('sonar.csv', 'Sonar'),
    ('spambase.csv', 'Spambase'),
    ('vehicle.csv', 'Vehicle'),
    ('waveform.csv', 'Waveform'),
    ('wine.csv', 'Wine'),
    ('winequality-red.csv', 'Wine Quality'),
    ('yeast.csv', 'Yeast'),
]

print(f'Running {len(datasets)} datasets...')
summary = []

for fname, dname in datasets:
    path = DATA_DIR / fname
    if not path.exists():
        print(f'  [{dname}] file not found')
        continue

    try:
        X, y = load_prepare(path)
    except Exception as e:
        print(f'  [{dname}] load error: {e}')
        continue

    n = len(X); nf = X.shape[1]; nc = len(np.unique(y))
    if nc < 2:
        print(f'  [{dname}] only {nc} class(es)')
        continue

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED)

    scaler = MinMaxScaler((-1, 1))
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # SVM
    t0 = time.time()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=SEED)
    svm.fit(X_tr_s, y_tr)
    svm_pred = svm.predict(X_te_s)
    svm_t = time.time() - t0
    svm_acc = accuracy_score(y_te, svm_pred)

    # JEPA+SVM
    emb = max(8, min(nf * 2, 32))
    pipe = JEPASVMEnsemble(random_state=SEED, embedding_dim=emb)
    t0 = time.time()
    res = pipe.run_pipeline(X_tr, X_te, y_tr, y_te)
    jepa_t = time.time() - t0

    jepa_pred = np.array(res['predictions'])
    jepa_acc = accuracy_score(y_te, jepa_pred)

    impr = ((jepa_acc - svm_acc) / svm_acc * 100) if svm_acc > 0 else 0
    if impr > 0:
        best = 'JEPA'
    elif impr == 0:
        best = 'Tie'
    else:
        best = 'SVM'

    summary.append((dname, n, nf, nc, svm_acc * 100, jepa_acc * 100, impr, best, jepa_t))
    print(f'  {dname:<20} n={n:<5} SVM={svm_acc*100:.2f}% JEPA={jepa_acc*100:.2f}% impr={impr:+.2f}% {best} [{jepa_t:.0f}s]')

# Summary table
print()
sep = '-' * 62
print(sep)
print(f'  {"Dataset":<20} {"n":<5} {"SVM":<10} {"JEPA":<10} {"Impr":<8} Best')
print(sep)
avg_svm = np.mean([r[4] for r in summary]) / 100
avg_jepa = np.mean([r[5] for r in summary]) / 100
for r in summary:
    print(f'  {r[0]:<20} {r[1]:<5} {r[4]:<9.2f}% {r[5]:<9.2f}% {r[6]:+.2f}% {r[7]:>6}')
print(sep)
avg_impr = ((avg_jepa - avg_svm) / avg_svm * 100) if avg_svm > 0 else 0
print(f'  {"Average":<20} {"":<5} {avg_svm*100:<9.2f}% {avg_jepa*100:<9.2f}% {avg_impr:+.2f}%')
wins = sum(1 for r in summary if r[7] == 'JEPA')
ties = sum(1 for r in summary if r[7] == 'Tie')
losses = sum(1 for r in summary if r[7] == 'SVM')
print(f'  Wins={wins}/{len(summary)} Ties={ties}/{len(summary)} Losses={losses}/{len(summary)}')
print(f'  No regression: {"YES" if losses == 0 else "CHECK"}')
