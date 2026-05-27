"""
EBM-SVM Trên Adult Dataset (Income Prediction)

So sánh:
- SVM Baseline + Grid Search: ~84% (theo bài báo)
- EBM-SVM: > 84% (mục tiêu)
"""

import sys
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Thêm đường dẫn
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.adult_data_processor import AdultDataProcessor
from ml_models.services.optimized_pipeline import OptimizedEBMSVMPipeline
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_ebm_svm_adult():
    """Train EBM-SVM trên Adult dataset."""
    
    logger.info("\n" + "="*80)
    logger.info("EBM-SVM TRÊN ADULT DATASET (Income Prediction)")
    logger.info("="*80)
    logger.info("Mục tiêu: Vượt quá SVM baseline (84%)")
    logger.info("="*80 + "\n")
    
    try:
        # 1. Tải & Tiền xử lý
        logger.info("[Bước 1] Tải & Tiền Xử Lý Adult Dataset")
        logger.info("-" * 80)
        
        processor = AdultDataProcessor()
        df_train, df_test, columns = processor.download_adult_dataset()
        X_train, X_test, y_train, y_test, scaler = processor.preprocess_adult_data(df_train, df_test)
        
        baseline_acc = 0.84  # Từ bài báo
        
        # 2. Train SVM Baseline (PSO Optimization - nhanh hơn)
        logger.info("\n[Bước 2] Train SVM Baseline (RBF + PSO Optimization)")
        logger.info("-" * 80)
        
        # Import RobustSVMTuner
        from ml_models.robust_svm_tuner import RobustSVMTuner
        
        tuner = RobustSVMTuner(cv_folds=5)
        tuning_result = tuner.tune_with_ensemble(
            X_train, y_train,
            c_range=(0.1, 1000),
            gamma_range=(1e-4, 1e-1),
            use_pso=True,
            n_pso_starts=2  # Chỉ 2 runs để nhanh hơn
        )
        
        best_svm = tuner.best_model
        y_pred_baseline = best_svm.predict(tuner.scaler.transform(X_test))
        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        
        logger.info(f"\n  Best params: {tuning_result['best_params']}")
        logger.info(f"  SVM Baseline Accuracy: {acc_baseline:.4f} ({acc_baseline*100:.2f}%)")
        logger.info(f"  Bài báo mục tiêu: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        
        # 3. Train EBM-SVM
        logger.info("\n[Bước 3] Train EBM-SVM")
        logger.info("-" * 80)
        logger.info("Cấu Hình EBM:")
        logger.info("  - Embedding Dim: 64")
        logger.info("  - Hidden Dim: 256")
        logger.info("  - Epochs: 800")
        logger.info("  - Learning Rate: 0.001")
        
        # Cấu hình EBM tối ưu
        ebm_config = {
            "hidden_dim": 256,
            "embedding_dim": 64,
            "epochs": 800,
            "learning_rate": 0.001,
            "noise_scale": 3.0
        }
        
        # Train EBM embeddings
        logger.info("\n  Huấn luyện EBM...")
        
        import torch
        from ml_models.ebm_svm import EBMEncoder, EBMTrainer, get_embeddings
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        
        ebm = EBMEncoder(
            input_dim=X_train.shape[1],
            hidden_dim=ebm_config["hidden_dim"],
            embedding_dim=ebm_config["embedding_dim"]
        )
        
        trainer = EBMTrainer(
            ebm,
            epochs=ebm_config["epochs"],
            learning_rate=ebm_config["learning_rate"],
            noise_scale=ebm_config["noise_scale"]
        )
        
        ebm = trainer.train(X_train_tensor, verbose=False)
        
        # Trích embeddings
        train_embeddings = get_embeddings(ebm, X_train_tensor)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_embeddings = get_embeddings(ebm, X_test_tensor)
        
        logger.info(f"  ✓ EBM training done. Embeddings shape: {train_embeddings.shape}")
        
        # 4. Tạo enhanced features
        logger.info("\n  Tạo enhanced features...")
        
        # Chỉ dùng top embeddings
        embedding_var = np.var(train_embeddings, axis=0)
        top_indices = np.argsort(embedding_var)[-32:]  # Top 32 embeddings
        top_train_embeddings = train_embeddings[:, top_indices]
        top_test_embeddings = test_embeddings[:, top_indices]
        
        # Kết hợp
        X_train_enhanced = np.hstack((X_train, top_train_embeddings))
        X_test_enhanced = np.hstack((X_test, top_test_embeddings))
        
        logger.info(f"  ✓ Enhanced features shape: {X_train_enhanced.shape}")
        
        # 5. Train SVM trên enhanced features
        logger.info("\n  Train SVM trên enhanced features (Grid Search)...")
        
        svm_enhanced = SVC(kernel='rbf')
        grid_search_enhanced = GridSearchCV(svm_enhanced, param_grid, cv=5, n_jobs=-1, verbose=0)
        grid_search_enhanced.fit(X_train_enhanced, y_train)
        
        best_svm_enhanced = grid_search_enhanced.best_estimator_
        y_pred_enhanced = best_svm_enhanced.predict(X_test_enhanced)
        acc_enhanced = accuracy_score(y_test, y_pred_enhanced)
        
        logger.info(f"  ✓ Best params: {grid_search_enhanced.best_params_}")
        logger.info(f"  EBM-SVM Accuracy: {acc_enhanced:.4f} ({acc_enhanced*100:.2f}%)")
        
        # 6. So Sánh Kết Quả
        logger.info("\n" + "="*80)
        logger.info("KẾT QUẢ SO SÁNH")
        logger.info("="*80)
        
        improvement = acc_enhanced - acc_baseline
        improvement_pct = (improvement / acc_baseline) * 100
        
        logger.info(f"\nĐộ Chính Xác:")
        logger.info(f"  SVM Baseline:       {acc_baseline:.4f} ({acc_baseline*100:.2f}%)")
        logger.info(f"  EBM-SVM:            {acc_enhanced:.4f} ({acc_enhanced*100:.2f}%)")
        logger.info(f"  Cải Thiện:          +{improvement:.4f} (+{improvement_pct:.2f}%)")
        
        logger.info(f"\nSo Sánh Với Bài Báo:")
        logger.info(f"  Bài Báo (SVM):      ~84% (84.00%)")
        logger.info(f"  Baseline Chúng Ta:  {acc_baseline*100:.2f}%")
        logger.info(f"  EBM-SVM Chúng Ta:   {acc_enhanced*100:.2f}%")
        
        if acc_enhanced > 0.84:
            logger.info(f"\n✓ THÀNH CÔNG! Vượt quá 84% của bài báo!")
            logger.info(f"  Đạt được: {acc_enhanced*100:.2f}%")
        else:
            logger.info(f"\n⚠️  Chưa vượt quá 84%")
            logger.info(f"  Hiện tại: {acc_enhanced*100:.2f}%")
        
        # 7. Chi Tiết Metrics
        logger.info(f"\n{'-'*80}")
        logger.info("Classification Report (EBM-SVM):")
        logger.info(f"{'-'*80}")
        logger.info(f"\n{classification_report(y_test, y_pred_enhanced)}")
        
        logger.info(f"\n{'-'*80}")
        logger.info("Confusion Matrix (EBM-SVM):")
        logger.info(f"{'-'*80}")
        cm = confusion_matrix(y_test, y_pred_enhanced)
        logger.info(f"\n{cm}")
        
        logger.info("\n" + "="*80)
        
        return {
            "svm_baseline": acc_baseline,
            "ebm_svm": acc_enhanced,
            "improvement": improvement,
            "improvement_pct": improvement_pct
        }
        
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = train_ebm_svm_adult()
    
    if results:
        sys.exit(0 if results["ebm_svm"] > 0.84 else 1)
    else:
        sys.exit(1)
