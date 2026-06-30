"""
Run baseline classifiers on all 23 UCI datasets (5-Fold CV)
Classifiers: SVM (RBF grid search), XGBoost, Random Forest
Output: results/baselines.json
"""
import sys, time, warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb

DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
N_FOLDS = 5

datasets = [
    ('abalone.csv', 'Abalone'), ('balance.csv', 'Balance'),
    ('banknote.csv', 'Banknote'), ('breast-cancer.csv', 'Breast Cancer'),
    ('car.csv', 'Car'), ('dermatology.csv', 'Dermatology'),
    ('ecoli.csv', 'Ecoli'), ('glass.csv', 'Glass'),
    ('haberman.csv', 'Haberman'), ('heart.csv', 'Heart'),
    ('ionosphere.csv', 'Ionosphere'), ('iris.csv', 'Iris'),
    ('liver.csv', 'Liver'), ('mushroom.csv', 'Mushroom'),
    ('optical.csv', 'Optical'), ('page-blocks.csv', 'Page Blocks'),
    ('sonar.csv', 'Sonar'), ('spambase.csv', 'Spambase'),
    ('vehicle.csv', 'Vehicle'), ('waveform.csv', 'Waveform'),
    ('wine.csv', 'Wine'), ('winequality-red.csv', 'Wine Quality'),
    ('yeast.csv', 'Yeast'),
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

def run_svm(X, y, fold_accs, fold_metrics):
    use_stratified = min(np.bincount(y)) >= N_FOLDS
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED) if use_stratified \
         else KFold(N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in kf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        param_grid = {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 'auto', 0.01, 0.1]}
        gs = GridSearchCV(SVC(kernel='rbf', random_state=SEED), param_grid,
                          cv=min(3, len(np.unique(y_tr))), scoring='accuracy', n_jobs=1)
        gs.fit(X_tr_s, y_tr)
        y_pred = gs.predict(X_te_s)
        fold_accs.append(accuracy_score(y_te, y_pred))
        fold_metrics.append({
            'accuracy': accuracy_score(y_te, y_pred),
            'predictions': y_pred.tolist(),
            'y_true': y_te.tolist(),
        })

def run_xgb(X, y, fold_accs, fold_metrics):
    use_stratified = min(np.bincount(y)) >= N_FOLDS
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED) if use_stratified \
         else KFold(N_FOLDS, shuffle=True, random_state=SEED)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
    }
    for train_idx, test_idx in kf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        n_classes = len(np.unique(y_tr))
        objective = 'multi:softmax' if n_classes > 2 else 'binary:logistic'
        eval_metric = 'mlogloss' if n_classes > 2 else 'logloss'
        model = xgb.XGBClassifier(objective=objective, eval_metric=eval_metric,
                                   random_state=SEED, n_jobs=1, verbosity=0)
        gs = GridSearchCV(model, param_grid, cv=min(3, n_classes),
                          scoring='accuracy', n_jobs=1)
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)
        fold_accs.append(accuracy_score(y_te, y_pred))
        fold_metrics.append({
            'accuracy': accuracy_score(y_te, y_pred),
            'predictions': y_pred.tolist(),
            'y_true': y_te.tolist(),
        })

def run_rf(X, y, fold_accs, fold_metrics):
    use_stratified = min(np.bincount(y)) >= N_FOLDS
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED) if use_stratified \
         else KFold(N_FOLDS, shuffle=True, random_state=SEED)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    for train_idx, test_idx in kf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = RandomForestClassifier(random_state=SEED, n_jobs=1)
        gs = GridSearchCV(model, param_grid, cv=min(3, len(np.unique(y_tr))),
                          scoring='accuracy', n_jobs=1)
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)
        fold_accs.append(accuracy_score(y_te, y_pred))
        fold_metrics.append({
            'accuracy': accuracy_score(y_te, y_pred),
            'predictions': y_pred.tolist(),
            'y_true': y_te.tolist(),
        })

def compute_metrics(fold_accs, fold_metrics):
    acc_mean = float(np.mean(fold_accs)) * 100
    acc_std = float(np.std(fold_accs)) * 100
    last = fold_metrics[-1]
    p = precision_recall_fscore_support(last['y_true'], last['predictions'],
                                         average='macro', zero_division=0)
    f1w = precision_recall_fscore_support(last['y_true'], last['predictions'],
                                           average='weighted', zero_division=0)
    return {
        'accuracy': round(acc_mean, 2),
        'accuracy_std': round(acc_std, 2),
        'precision_macro': round(p[0] * 100, 2),
        'recall_macro': round(p[1] * 100, 2),
        'f1_macro': round(p[2] * 100, 2),
        'f1_weighted': round(f1w[2] * 100, 2),
    }

results = {}
for classifier_name, run_fn in [('SVM', run_svm), ('XGBoost', run_xgb), ('RandomForest', run_rf)]:
    print(f'\n{"="*60}')
    print(f'Running {classifier_name} on {len(datasets)} datasets...')
    print(f'{"="*60}')
    classifier_results = []
    for idx, (fname, dname) in enumerate(datasets):
        path = DATA_DIR / fname
        if not path.exists():
            print(f'[{dname:<22}] NOT FOUND')
            continue
        try:
            X, y = load_and_prepare(path)
        except Exception as e:
            print(f'[{dname:<22}] Load fail: {str(e)[:80]}')
            classifier_results.append({'dataset': dname, 'error': str(e)[:200]})
            continue
        n, nf, nc = len(X), X.shape[1], len(np.unique(y))
        if nc < 2:
            print(f'[{dname:<22}] Skip: {nc} class(es)')
            continue
        t0 = time.time()
        fold_accs, fold_metrics = [], []
        try:
            run_fn(X, y, fold_accs, fold_metrics)
            m = compute_metrics(fold_accs, fold_metrics)
            m.update({'dataset': dname, 'samples': n, 'features': nf, 'classes': nc,
                       'time_seconds': round(time.time() - t0, 1)})
            classifier_results.append(m)
            print(f'[{dname:<22}] {n:>5}x{nf:<4} {nc}cls  Acc={m["accuracy"]:.2f}% ±{m["accuracy_std"]:.2f}')
        except Exception as e:
            err = str(e).split('\n')[0][:150]
            print(f'[{dname:<22}] ERROR: {err}')
            classifier_results.append({'dataset': dname, 'error': err})
    results[classifier_name] = classifier_results

# Save baselines
output = {'results': results, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
with open(RESULTS_DIR / 'baselines.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\nSaved {RESULTS_DIR / "baselines.json"}')

# Summary
for name, res in results.items():
    success = [r for r in res if 'error' not in r]
    if success:
        avg_acc = np.mean([r['accuracy'] for r in success])
        print(f'{name:<15}  Avg Acc: {avg_acc:.2f}%  ({len(success)}/{len(res)} datasets)')
