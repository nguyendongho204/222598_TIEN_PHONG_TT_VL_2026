"""
Batch train JEPA+SVM on all UCI datasets (K-Fold CV)
Records: accuracy, precision, recall, F1 (macro & weighted)
Output: results/ketqua.txt
Match web API: OneHotEncoder, K-Fold 5, embedding_dim=32
"""
import sys, time, warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

api_dir = str(Path(__file__).parent.parent / 'api_base')
ml_dir = str(Path(__file__).parent.parent / 'api_base' / 'ml_models')
for p in [api_dir, ml_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_models.jepa_svm_pipeline import JEPASVMEnsemble

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
    """OneHotEncoder cho categorical, giữ nguyên numeric (giống web API)."""
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
    X_raw = X.values

    X_encoded = encode_features(X_raw)
    return X_encoded, y


def save_all(results, total_time):
    lines, success, failed = [], [r for r in results if 'error' not in r], [r for r in results if 'error' in r]

    lines.append('=' * 130)
    lines.append('KET QUA THUC NGHIEM - JEPA+SVM (5-Fold CV)')
    lines.append('=' * 130)
    lines.append(f'Dataset: {len(success)}/{len(results)} thanh cong')
    lines.append(f'Thoi gian: {total_time:.0f}s ({total_time/60:.1f} phut)')
    lines.append('')

    if success:
        lines.append('-' * 130)
        h = f'{"Dataset":<22} {"n":<7} {"Feat":<5} {"Cls":<5} {"Acc":<9} {"Std":<7} {"Prec(m)":<10} {"Rec(m)":<10} {"F1(m)":<10} {"F1(w)":<10} {"Time":<7}'
        lines.append(h)
        lines.append('-' * 130)
        for r in sorted(success, key=lambda x: x['dataset']):
            acc_str = f'{r["accuracy"]:<6.2f}%' if r["accuracy_std"] is None else f'{r["accuracy"]:<6.2f}%'
            std_str = f'±{r["accuracy_std"]:.2f}' if r["accuracy_std"] is not None else ''
            lines.append(f'{r["dataset"]:<22} {r["samples"]:<7} {r["features"]:<5} {r["classes"]:<5} {acc_str} {std_str:<7} {r["precision_macro"]:<9.2f}% {r["recall_macro"]:<9.2f}% {r["f1_macro"]:<9.2f}% {r["f1_weighted"]:<9.2f}% {r["time_seconds"]:<6.0f}s')
        lines.append('-' * 130)
        avg = lambda k: np.mean([r[k] for r in success])
        lines.append(f'{"TRUNG BINH":<22} {"":<7} {"":<5} {"":<5} {avg("accuracy"):<6.2f}% {"":<7} {avg("precision_macro"):<9.2f}% {avg("recall_macro"):<9.2f}% {avg("f1_macro"):<9.2f}% {avg("f1_weighted"):<9.2f}%')
        lines.append('')

    if failed:
        lines.append('DATASET LOI:')
        for r in failed:
            lines.append(f'  {r["dataset"]:<22} {r.get("error", "")[:120]}')

    (RESULTS_DIR / 'ketqua.txt').write_text('\n'.join(lines), encoding='utf-8')

    clean = [{'dataset': r['dataset'], 'samples': r['samples'], 'features': r['features'],
              'classes': r['classes'], 'accuracy': r['accuracy'],
              'accuracy_std': r['accuracy_std'],
              'precision_macro': r['precision_macro'], 'recall_macro': r['recall_macro'],
              'f1_macro': r['f1_macro'], 'f1_weighted': r['f1_weighted'],
              'time_seconds': r['time_seconds']} for r in success]
    with open(RESULTS_DIR / 'ketqua.json', 'w', encoding='utf-8') as f:
        json.dump({'results': clean, 'failed': failed, 'total_time': total_time}, f, indent=2, ensure_ascii=False)
    print(f'\nSaved {RESULTS_DIR / "ketqua.txt"} ({len(success)} datasets)')


print(f'Training {len(datasets)} datasets (K-Fold={N_FOLDS})...')
results, total_start = [], time.time()

for idx, (fname, dname) in enumerate(datasets):
    path = DATA_DIR / fname
    if not path.exists():
        print(f'[{dname:<22}] NOT FOUND')
        continue

    try:
        X, y = load_and_prepare(path)
    except Exception as e:
        print(f'[{dname:<22}] Load fail: {str(e)[:80]}')
        results.append({'dataset': dname, 'error': str(e)[:200]})
        continue

    n, nf, nc = len(X), X.shape[1], len(np.unique(y))
    if nc < 2:
        print(f'[{dname:<22}] Skip: {nc} class(es)')
        continue

    try:
        t0 = time.time()

        use_stratified = min(np.bincount(y)) >= N_FOLDS
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED) if use_stratified \
             else KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        fold_accs, fold_preds_all = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            res = JEPASVMEnsemble(random_state=SEED + fold_idx, embedding_dim=32).run_pipeline(
                X_tr, X_te, y_tr, y_te)
            fold_accs.append(res['accuracy'])
            fold_preds_all.append(np.array(res['predictions']))

        # Dùng fold cuối để tính metrics chi tiết
        y_pred = fold_preds_all[-1]
        p_macro = precision_score(y[test_idx], y_pred, average='macro', zero_division=0) * 100
        r_macro = recall_score(y[test_idx], y_pred, average='macro', zero_division=0) * 100
        f1_macro = f1_score(y[test_idx], y_pred, average='macro', zero_division=0) * 100
        f1_weighted = f1_score(y[test_idx], y_pred, average='weighted', zero_division=0) * 100

        acc_mean = float(np.mean(fold_accs)) * 100
        acc_std = float(np.std(fold_accs)) * 100

        elapsed = time.time() - t0

        results.append({
            'dataset': dname, 'samples': n, 'features': nf, 'classes': nc,
            'accuracy': round(acc_mean, 2),
            'accuracy_std': round(acc_std, 2),
            'precision_macro': round(p_macro, 2), 'recall_macro': round(r_macro, 2),
            'f1_macro': round(f1_macro, 2), 'f1_weighted': round(f1_weighted, 2),
            'time_seconds': round(elapsed, 1),
        })
        print(f'[{dname:<22}] {n:>5}x{nf:<4} {nc}cls  Acc={acc_mean:.2f}% ±{acc_std:.2f}  F1(m)={f1_macro:.2f}%  [{elapsed:.0f}s]')

    except Exception as e:
        err = str(e).split('\n')[0][:150]
        print(f'[{dname:<22}] ERROR: {err}')
        results.append({'dataset': dname, 'error': err})

    if (idx + 1) % 5 == 0 or idx == len(datasets) - 1:
        save_all(results, time.time() - total_start)

total_time = time.time() - total_start
save_all(results, total_time)
success = [r for r in results if 'error' not in r]
print(f'\nDone: {len(success)}/{len(results)} datasets')
if success:
    print(f'Avg Acc={np.mean([r["accuracy"] for r in success]):.2f}%  F1(macro)={np.mean([r["f1_macro"] for r in success]):.2f}%')
