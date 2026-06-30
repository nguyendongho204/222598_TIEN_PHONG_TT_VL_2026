"""
Download and standardize ~20 classification datasets from UCI ML Repository.

Output: data/{name}.csv
Format: All features + 'label' column (last column = target)
"""

import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'api_base'))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path('D:/Nam4/ThucTap/EBM_SVM/data')
DATA_DIR.mkdir(exist_ok=True)

# Dataset definitions: (filename, uci_id, target_col_pattern)
# target_col_pattern = prefix or full name of target column(s)
DATASETS = [
    # Already have - skip download but list for benchmark
    ('iris',          53,   None),
    ('wine',          109,  None),
    ('breast-cancer', 17,   None),
    ('winequality-red', 186, None),
    ('adult',         2,    None),
    ('diabetes',      891,  None),

    # New datasets to download
    ('heart',         45,   None),
    ('sonar',         151,  None),
    ('ionosphere',    52,   None),
    ('glass',         42,   None),
    ('yeast',         110,  None),
    ('ecoli',         39,   None),
    ('vehicle',       149,  None),
    ('abalone',       1,    None),
    ('car',           19,   None),
    ('haberman',      43,   None),
    ('liver',         60,   None),
    ('balance',       12,   None),
    ('dermatology',   33,   None),
    ('banknote',      267,  None),
    ('optical',       80,   None),
    ('spambase',      94,   None),
    ('mushroom',      73,   None),
    ('page-blocks',   78,   None),
    ('waveform',      107,  None),
]

# Datasets that are already prepared as CSV
EXISTING_CSV = {
    'iris', 'wine', 'breast-cancer', 'winequality-red', 'adult', 'diabetes'
}


def infer_target_col(df, metadata=None):
    """Find the most likely target column."""
    candidates = []
    if metadata is not None:
        # ucimlrepo provides target columns info
        if hasattr(metadata, 'target_col') and metadata.target_col:
            for col in metadata.target_col:
                if col in df.columns:
                    candidates.append(col)
        if hasattr(metadata, 'original_id') and metadata.original_id:
            logger.info(f"  UCI ID: {metadata.original_id}")

    # Heuristic: columns with few unique values
    for col in df.columns:
        if col in candidates:
            continue
        if df[col].dtype == 'object' or df[col].nunique() <= 20:
            if df[col].nunique() >= 2 and df[col].nunique() <= 20:
                candidates.append(col)

    if not candidates:
        return df.columns[-1]

    # Prefer column named 'class', 'target', 'label', 'Class', etc.
    for name in ['class', 'target', 'label', 'Class', 'Target', 'Label',
                 'CLASS', 'type', 'Type', 'category']:
        for c in candidates:
            if c.lower() == name.lower():
                return c

    # Prefer columns with fewer unique values (more likely to be target)
    candidates.sort(key=lambda c: (df[c].nunique(), c != df.columns[-1]))

    # Prefer last column if it's in candidates
    if df.columns[-1] in candidates:
        return df.columns[-1]

    return candidates[0] if candidates else df.columns[-1]


def download_and_save(name, uci_id):
    """Download a dataset from UCI and save as standardized CSV."""
    output_path = DATA_DIR / f"{name}.csv"
    if output_path.exists():
        logger.info(f"  [SKIP] {name}.csv already exists")
        return True

    logger.info(f"  Downloading '{name}' (ID={uci_id})...")
    try:
        repo = fetch_ucirepo(id=uci_id)
        df = repo.data.original
        metadata = repo.metadata

        if df is None:
            # Try combined features + targets
            X = repo.data.features
            y = repo.data.targets
            if X is None or y is None:
                logger.warning(f"  [FAIL] No data for '{name}'")
                return False
            df = pd.concat([X, y], axis=1)

        logger.info(f"  Shape: {df.shape}")

        # Handle multi-column target: take first target column
        target_col = infer_target_col(df, metadata)
        logger.info(f"  Target: '{target_col}'")

        if target_col not in df.columns:
            # Try to find it case-insensitive
            for c in df.columns:
                if c.lower() == target_col.lower():
                    target_col = c
                    break

        # Rearrange: features + target as last column named 'label'
        feature_cols = [c for c in df.columns if c != target_col]
        X_df = df[feature_cols].copy()
        y_df = df[target_col].copy()

        # Drop any ID/index columns
        id_cols = [c for c in X_df.columns if c.lower() in ['id', 'ids', 'index', 'sample']]
        X_df = X_df.drop(columns=id_cols, errors='ignore')

        # Encode any object columns
        for c in X_df.select_dtypes(include=['object']).columns:
            X_df[c] = pd.factorize(X_df[c])[0]

        # Encode target (luôn factorize để đảm bảo là int 0..n_classes-1)
        y_encoded = pd.factorize(y_df)[0]

        # Combine
        result = X_df.copy()
        result['label'] = y_encoded

        result.to_csv(output_path, index=False)
        logger.info(f"  [OK] Saved {name}.csv ({result.shape[0]} samples, {result.shape[1]-1} features)")
        return True

    except Exception as e:
        logger.error(f"  [FAIL] {name}: {str(e)}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("UCI DATASET DOWNLOADER")
    logger.info("=" * 60)

    success = 0
    skip = 0
    fail = 0

    for name, uci_id, _ in DATASETS:
        if name in EXISTING_CSV:
            logger.info(f"\n  [EXISTS] {name}.csv (already in data folder)")
            skip += 1
            continue

        logger.info("")
        ok = download_and_save(name, uci_id)
        if ok:
            if (DATA_DIR / f"{name}.csv").exists():
                success += 1
            else:
                fail += 1
        else:
            fail += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"SUMMARY: {success} downloaded, {skip} existing, {fail} failed")
    logger.info("=" * 60)

    # List all datasets
    logger.info("\nAvailable datasets in data/:")
    for f in sorted(DATA_DIR.glob('*.csv')):
        df = pd.read_csv(f, nrows=0)
        logger.info(f"  {f.stem:<20} ({f.stat().st_size/1024:>7.0f} KB, {len(df.columns):>3} cols)")


if __name__ == '__main__':
    main()
