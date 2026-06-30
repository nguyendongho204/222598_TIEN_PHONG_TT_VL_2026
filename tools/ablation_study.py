"""
Ablation Study: JEPA+SVM component analysis
Configurations:
  1. Full: self-supervised + supervised fine-tune + SVM embeddings
  2. NoSSL: supervised fine-tune only (skip self-supervised) + SVM embeddings
  3. NoFT: self-supervised only (skip supervised fine-tune) + SVM embeddings
  4. SVM-only: raw features + SVM grid search (no JEPA)

Runs on 10 representative UCI datasets.
Output: results/ablation.json
"""
import sys, time, warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

api_dir = str(Path(__file__).parent.parent / 'api_base')
ml_dir = str(Path(__file__).parent.parent / 'api_base' / 'ml_models')
for p in [api_dir, ml_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_models.jepa_model import JEPA

DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
N_FOLDS = 5

# 10 representative datasets covering easy/medium/hard, small/large
ABLATION_DATASETS = [
    ('abalone.csv', 'Abalone'),       # hard, many classes
    ('breast-cancer.csv', 'Breast Cancer'),  # easy, small
    ('car.csv', 'Car'),               # categorical, medium
    ('ecoli.csv', 'Ecoli'),           # medium, multiclass
    ('glass.csv', 'Glass'),           # small, multiclass
    ('haberman.csv', 'Haberman'),     # hard, binary
    ('ionosphere.csv', 'Ionosphere'), # medium, binary
    ('optical.csv', 'Optical'),       # easy, large, multiclass
    ('spambase.csv', 'Spambase'),     # medium, large, binary
    ('winequality-red.csv', 'Wine Quality'),  # hard, medium
]

def encode_features(X_raw):
    num_cols, cat_cols = [], []
    for col_idx in range(X_raw.shape[1]):
        try:
            X_raw[:, col_idx].astype(float)
            num_cols.append(col_idx)
        except (ValueError, TypeError):
            cat_cols.append(col_idx)
    parts = []
    if num_cols:
        num_part = np.zeros((X_raw.shape[0], len(num_cols)), dtype=np.float32)
        for i, col_idx in enumerate(num_cols):
            col = X_raw[:, col_idx]
            arr = np.array([float(v) for v in col], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=np.nanmedian(arr) if not np.isnan(np.nanmedian(arr)) else 0.0)
            num_part[:, i] = arr
        parts.append(num_part)
    if cat_cols:
        X_cat = X_raw[:, cat_cols].astype(str)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        parts.append(ohe.fit_transform(X_cat))
    return np.hstack(parts).astype(np.float32) if parts else np.zeros((X_raw.shape[0], 0))

def load_and_prepare(path):
    df = pd.read_csv(path)
    target = 'label' if 'label' in df.columns else df.columns[-1]
    if 'winequality' in str(path).lower():
        y = (df[target].values >= 6).astype(int)
    else:
        y = LabelEncoder().fit_transform(df[target].astype(str).str.strip())
    drop_cols = [target] + [c for c in df.columns if c.lower() in ['id', 'ids']]
    X = df.drop(columns=drop_cols, errors='ignore').copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X_encoded = encode_features(X.values)
    return X_encoded, y

def _adjust_epochs(n_samples):
    if n_samples < 300:
        return 600
    elif n_samples < 800:
        return 500
    elif n_samples < 2000:
        return 300
    return 200

# === Configuration runners ===

def run_full_jepa_svm(X_tr, X_te, y_tr, y_te):
    """Self-supervised + supervised fine-tune + SVM on embeddings"""
    scaler = MinMaxScaler((-1, 1))
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    epochs = _adjust_epochs(len(X_tr))
    jepa = JEPA(input_dim=X_tr_s.shape[1], embedding_dim=32, hidden_dims=[64, 32], device='cpu')
    jepa.train(X_tr_s, epochs=epochs, batch_size=min(32, len(X_tr_s)//2), mask_rate=0.3, lr=0.001, verbose=False)
    jepa.train_supervised(X_tr_s, y_tr, epochs=max(30, epochs//5), lr=0.0005, verbose=False)

    train_emb = jepa.extract_features(X_tr_s)
    test_emb = jepa.extract_features(X_te_s)

    svm = SVC(kernel='rbf', random_state=SEED, probability=True)
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto', 0.01, 0.1]}
    gs = GridSearchCV(svm, param_grid, cv=min(3, len(np.unique(y_tr))), scoring='accuracy', n_jobs=1)
    gs.fit(train_emb, y_tr)
    return gs.predict(test_emb)

def run_no_ssl_jepa_svm(X_tr, X_te, y_tr, y_te):
    """Skip self-supervised: init encoder random, just supervised fine-tune + SVM"""
    scaler = MinMaxScaler((-1, 1))
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    jepa = JEPA(input_dim=X_tr_s.shape[1], embedding_dim=32, hidden_dims=[64, 32], device='cpu')
    jepa.train_supervised(X_tr_s, y_tr, epochs=max(30, _adjust_epochs(len(X_tr))//5), lr=0.0005, verbose=False)

    train_emb = jepa.extract_features(X_tr_s)
    test_emb = jepa.extract_features(X_te_s)

    svm = SVC(kernel='rbf', random_state=SEED, probability=True)
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto', 0.01, 0.1]}
    gs = GridSearchCV(svm, param_grid, cv=min(3, len(np.unique(y_tr))), scoring='accuracy', n_jobs=1)
    gs.fit(train_emb, y_tr)
    return gs.predict(test_emb)

def run_no_ft_jepa_svm(X_tr, X_te, y_tr, y_te):
    """Skip supervised fine-tune: self-supervised only + SVM on embeddings"""
    scaler = MinMaxScaler((-1, 1))
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    epochs = _adjust_epochs(len(X_tr))
    jepa = JEPA(input_dim=X_tr_s.shape[1], embedding_dim=32, hidden_dims=[64, 32], device='cpu')
    jepa.train(X_tr_s, epochs=epochs, batch_size=min(32, len(X_tr_s)//2), mask_rate=0.3, lr=0.001, verbose=False)

    train_emb = jepa.extract_features(X_tr_s)
    test_emb = jepa.extract_features(X_te_s)

    svm = SVC(kernel='rbf', random_state=SEED, probability=True)
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto', 0.01, 0.1]}
    gs = GridSearchCV(svm, param_grid, cv=min(3, len(np.unique(y_tr))), scoring='accuracy', n_jobs=1)
    gs.fit(train_emb, y_tr)
    return gs.predict(test_emb)

def run_svm_only(X_tr, X_te, y_tr, y_te):
    """Raw features + SVM grid search (no JEPA)"""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    svm = SVC(kernel='rbf', random_state=SEED, probability=True)
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto', 0.01, 0.1]}
    gs = GridSearchCV(svm, param_grid, cv=min(3, len(np.unique(y_tr))), scoring='accuracy', n_jobs=1)
    gs.fit(X_tr_s, y_tr)
    return gs.predict(X_te_s)

CONFIGS = {
    'Full_JEPA_SVM': run_full_jepa_svm,
    'No_SSL': run_no_ssl_jepa_svm,
    'No_FineTune': run_no_ft_jepa_svm,
    'SVM_Only': run_svm_only,
}

def run_one_config(config_name, run_fn, datasets_list):
    print(f'\n{"="*60}')
    print(f'Configuration: {config_name}')
    print(f'{"="*60}')
    config_results = []
    for idx, (fname, dname) in enumerate(datasets_list):
        path = DATA_DIR / fname
        if not path.exists():
            print(f'[{dname:<22}] NOT FOUND')
            continue
        try:
            X, y = load_and_prepare(path)
        except Exception as e:
            print(f'[{dname:<22}] Load fail: {str(e)[:80]}')
            config_results.append({'dataset': dname, 'error': str(e)[:200]})
            continue
        n, nf, nc = len(X), X.shape[1], len(np.unique(y))
        if nc < 2:
            print(f'[{dname:<22}] Skip: {nc} class(es)')
            continue
        t0 = time.time()
        fold_accs, fold_preds_all = [], []
        use_stratified = min(np.bincount(y)) >= N_FOLDS
        kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED) if use_stratified \
             else KFold(N_FOLDS, shuffle=True, random_state=SEED)
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                y_pred = run_fn(X_tr, X_te, y_tr, y_te)
                fold_accs.append(accuracy_score(y_te, y_pred))
                fold_preds_all.append(np.array(y_pred))

            y_pred_last = fold_preds_all[-1]
            y_test_last = y[test_idx]
            p = precision_recall_fscore_support(y_test_last, y_pred_last, average='macro', zero_division=0)
            f1w = precision_recall_fscore_support(y_test_last, y_pred_last, average='weighted', zero_division=0)
            acc_mean = float(np.mean(fold_accs)) * 100
            acc_std = float(np.std(fold_accs)) * 100
            elapsed = time.time() - t0
            config_results.append({
                'dataset': dname, 'samples': n, 'features': nf, 'classes': nc,
                'accuracy': round(acc_mean, 2), 'accuracy_std': round(acc_std, 2),
                'precision_macro': round(p[0] * 100, 2),
                'recall_macro': round(p[1] * 100, 2),
                'f1_macro': round(p[2] * 100, 2),
                'f1_weighted': round(f1w[2] * 100, 2),
                'time_seconds': round(elapsed, 1),
            })
            print(f'[{dname:<22}] Acc={acc_mean:.2f}% ±{acc_std:.2f}  F1={p[2]*100:.2f}%  [{elapsed:.0f}s]')
        except Exception as e:
            err = str(e).split('\n')[0][:150]
            print(f'[{dname:<22}] ERROR: {err}')
            config_results.append({'dataset': dname, 'error': err})
    return config_results

print('Ablation Study: JEPA+SVM Component Analysis')
print(f'{len(ABLATION_DATASETS)} datasets, {len(CONFIGS)} configurations, {N_FOLDS}-Fold CV')
all_results = {}

for config_name, run_fn in CONFIGS.items():
    all_results[config_name] = run_one_config(config_name, run_fn, ABLATION_DATASETS)

output = {
    'results': all_results,
    'datasets': [d[1] for d in ABLATION_DATASETS],
    'configs': list(CONFIGS.keys()),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}
with open(RESULTS_DIR / 'ablation.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\nSaved {RESULTS_DIR / "ablation.json"}')

print('\n' + '='*60)
print('ABLATION SUMMARY (Avg Accuracy %)')
print('='*60)
print(f'{"Dataset":<20}', end='')
for cfg in CONFIGS:
    print(f'{cfg:<18}', end='')
    continue
print()
print('-'*80)
for dname in [d[1] for d in ABLATION_DATASETS]:
    print(f'{dname:<20}', end='')
    for cfg_name in CONFIGS:
        res_list = all_results[cfg_name]
        match = [r for r in res_list if r.get('dataset') == dname and 'error' not in r]
        if match:
            print(f'{match[0]["accuracy"]:<8.2f}%  {"":8}', end='')
        else:
            print(f'{"ERR":<18}', end='')
    print()
print('-'*80)
print(f'{"AVERAGE":<20}', end='')
for cfg_name in CONFIGS:
    res_list = all_results[cfg_name]
    accs = [r['accuracy'] for r in res_list if 'error' not in r]
    if accs:
        print(f'{np.mean(accs):<8.2f}%  {"":8}', end='')
    else:
        print(f'{"N/A":<18}', end='')
print()
