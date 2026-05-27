"""
Adult Benchmark - Optimized EBM-SVM vs Standard SVM

So sánh:
1. Standard SVM (RBF + Grid Search)
2. Optimized EBM-SVM (Enhanced features + PSO tuning)

Target: Chứng minh EBM-SVM vượt 85% trên Adult
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml_models.adult_data_processor import AdultDataProcessor
from ml_models.svm_tuning import SVMTuner
from ml_models.optimized_pipeline_adult_v2 import OptimizedPipelineAdultV2


def run_adult_benchmark():
    """
    Chạy benchmark đầy đủ trên Adult dataset.
    """
    
    logger.info("\n" + "="*100)
    logger.info("ADULT DATASET BENCHMARK - OPTIMIZED EBM-SVM vs STANDARD SVM")
    logger.info("="*100)
    
    # =========================================================================
    # Step 1: Load & Preprocess Adult Dataset
    # =========================================================================
    logger.info("\n[Bước 1] Tải & Tiền Xử Lý Adult Dataset")
    logger.info("-" * 100)
    
    processor = AdultDataProcessor()
    
    try:
        X_train, y_train, X_test, y_test = processor.load_and_preprocess_adult()
        logger.info(f"✓ Tải thành công!")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Test: {X_test.shape}")
    except Exception as e:
        logger.error(f"❌ Lỗi tải Adult dataset: {e}")
        return
    
    # =========================================================================
    # Step 2: Baseline SVM (Standard)
    # =========================================================================
    logger.info("\n[Bước 2] Train Standard SVM (Baseline)")
    logger.info("-" * 100)
    
    tuner = SVMTuner()
    baseline_accuracy = tuner.tune_standard_svm(X_train, y_train, X_test, y_test)
    
    logger.info(f"✓ SVM Baseline Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    
    # =========================================================================
    # Step 3: Optimized EBM-SVM
    # =========================================================================
    logger.info("\n[Bước 3] Train Optimized EBM-SVM")
    logger.info("-" * 100)
    logger.info("Cấu hình tối ưu:")
    logger.info("  - Embedding dimension: 256 (↑ từ 64)")
    logger.info("  - Hidden dimension: 512 (↑ từ 256)")
    logger.info("  - Epochs: 1500 (↑ từ 800)")
    logger.info("  - Learning rate: 0.0005 (↓ từ 0.001)")
    logger.info("  - Tuning: PSO (thay vì Grid Search)")
    logger.info("  - Features: Original + Interactions + Top 128 Embeddings")
    
    pipeline = OptimizedPipelineAdultV2(use_pso_tuning=True)
    
    try:
        result = pipeline.train_full_pipeline(X_train, y_train, X_test, y_test)
        optimized_accuracy = result['accuracy']
    except Exception as e:
        logger.error(f"❌ Lỗi train EBM-SVM: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # Step 4: So Sánh Kết Quả
    # =========================================================================
    logger.info("\n" + "="*100)
    logger.info("RESULTS - SO SÁNH KẾT QUẢ")
    logger.info("="*100)
    
    improvement = optimized_accuracy - baseline_accuracy
    improvement_pct = (improvement / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
    
    logger.info(f"\n{'Model':<30} | {'Accuracy':<15} | {'Chỉ Số':<20}")
    logger.info("-" * 100)
    logger.info(f"{'Standard SVM (Baseline)':<30} | {baseline_accuracy:<14.4f} | {baseline_accuracy*100:<19.2f}%")
    logger.info(f"{'Optimized EBM-SVM':<30} | {optimized_accuracy:<14.4f} | {optimized_accuracy*100:<19.2f}%")
    logger.info("-" * 100)
    logger.info(f"{'IMPROVEMENT':<30} | {improvement:<14.4f} | {improvement_pct:<19.2f}%")
    
    logger.info("\n" + "="*100)
    
    # Đánh giá
    if optimized_accuracy > 0.85:
        logger.info("✓✓✓ THÀNH CÔNG! EBM-SVM vượt quá 85% trên Adult dataset! 🎉")
    elif optimized_accuracy > baseline_accuracy:
        logger.info(f"✓ THÀNH CÔNG! EBM-SVM cao hơn SVM baseline ({improvement_pct:.2f}% improvement)")
    else:
        logger.info(f"❌ EBM-SVM thấp hơn SVM baseline. Cần điều chỉnh lại.")
    
    logger.info("\nChi tiết:")
    logger.info(f"  - SVM Baseline:       {baseline_accuracy*100:.2f}%")
    logger.info(f"  - EBM-SVM Optimized:  {optimized_accuracy*100:.2f}%")
    logger.info(f"  - Cải thiện:          +{improvement*100:.2f}% ({improvement_pct:.2f}%)")
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"\n🔧 GPU được sử dụng: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"\n🔧 Chạy trên CPU")
    
    logger.info("\n" + "="*100 + "\n")
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'optimized_accuracy': optimized_accuracy,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }


if __name__ == "__main__":
    results = run_adult_benchmark()
    
    # Save results
    if results:
        results_file = Path(__file__).parent.parent.parent / "adult_benchmark_results.txt"
        with open(results_file, 'w') as f:
            f.write("ADULT DATASET BENCHMARK RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Standard SVM:       {results['baseline_accuracy']:.4f} ({results['baseline_accuracy']*100:.2f}%)\n")
            f.write(f"Optimized EBM-SVM:  {results['optimized_accuracy']:.4f} ({results['optimized_accuracy']*100:.2f}%)\n")
            f.write(f"Improvement:        {results['improvement']:.4f} ({results['improvement_pct']:.2f}%)\n")
        
        logger.info(f"✓ Kết quả đã lưu vào: {results_file}")
