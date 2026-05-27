"""
Demo - Chạy Universal EBM-SVM Pipeline Trên Nhiều Datasets

Chứng minh rằng thuật toán tự động tối ưu cho mỗi dataset.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml_models.universal_pipeline import UniversalEBMSVMPipeline

# Load datasets
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import LabelEncoder
import csv


def load_breast_cancer():
    """Load Breast Cancer dataset."""
    df = pd.read_csv('../data/breast-cancer.csv')
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, -1])
    X = df.iloc[:, :-1].values
    
    return X, y, "Breast Cancer (569 samples, 30 features)"


def load_wine_quality():
    """Load Wine Quality dataset."""
    df = pd.read_csv('../data/winequality-red.csv')
    
    # Make binary classification (good vs not good)
    y = (df.iloc[:, -1] >= 6).astype(int).values
    X = df.iloc[:, :-1].values
    
    return X, y, "Wine Quality (1599 samples, 11 features)"


def run_benchmark():
    """Run benchmark trên multiple datasets."""
    
    logger.info("\n" + "="*100)
    logger.info("UNIVERSAL EBM-SVM PIPELINE - MULTI-DATASET BENCHMARK")
    logger.info("="*100)
    
    # Datasets để test
    datasets = [
        # Built-in datasets
        (load_iris()[0], load_iris()[1], "Iris (150 samples, 4 features)"),
        (load_wine()[0], load_wine()[1], "Wine (178 samples, 13 features)"),
    ]
    
    # CSV datasets
    try:
        X, y, name = load_breast_cancer()
        datasets.append((X, y, name))
        logger.info(f"✓ Loaded {name}")
    except Exception as e:
        logger.warning(f"✗ Không tải được Breast Cancer: {e}")
    
    try:
        X, y, name = load_wine_quality()
        datasets.append((X, y, name))
        logger.info(f"✓ Loaded {name}")
    except Exception as e:
        logger.warning(f"✗ Không tải được Wine Quality: {e}")
    
    # Run pipeline trên mỗi dataset
    results_summary = []
    
    for idx, (X, y, name) in enumerate(datasets, 1):
        logger.info("\n" + "="*100)
        logger.info(f"DATASET {idx}: {name}")
        logger.info("="*100)
        
        try:
            pipeline = UniversalEBMSVMPipeline(device='cpu')
            result = pipeline.run_full_pipeline(X, y, test_size=0.2)
            
            results_summary.append({
                'dataset': name,
                'baseline': result['baseline'],
                'optimized': result['optimized'],
                'improvement': result['improvement'],
                'improvement_pct': result['improvement_pct'],
                'status': '✓' if result['optimized'] > result['baseline'] else '❌'
            })
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    logger.info("\n" + "="*100)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*100)
    
    logger.info(f"\n{'Dataset':<50} | {'Baseline':<12} | {'Optimized':<12} | {'Improvement':<12} | {'Status':<5}")
    logger.info("-" * 100)
    
    for res in results_summary:
        baseline = res['baseline'] * 100
        optimized = res['optimized'] * 100
        improvement = res['improvement_pct']
        status = res['status']
        
        logger.info(f"{res['dataset']:<50} | {baseline:<11.2f}% | {optimized:<11.2f}% | {improvement:<11.2f}% | {status:<5}")
    
    # Overall statistics
    if results_summary:
        improvements = [r['improvement_pct'] for r in results_summary]
        success_count = sum(1 for r in results_summary if r['status'] == '✓')
        
        logger.info("-" * 100)
        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Datasets tested: {len(results_summary)}")
        logger.info(f"  Success rate: {success_count}/{len(results_summary)} ({success_count*100//len(results_summary)}%)")
        logger.info(f"  Average improvement: {np.mean(improvements):.2f}%")
        logger.info(f"  Best improvement: {max(improvements):.2f}%")
        logger.info(f"  Worst improvement: {min(improvements):.2f}%")
    
    logger.info("\n" + "="*100 + "\n")


if __name__ == "__main__":
    run_benchmark()
