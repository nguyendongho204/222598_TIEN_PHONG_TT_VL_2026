"""
Chạy PSO-SVM Tuning Trên Breast Cancer Dataset

Mục tiêu: Đạt 99%+ độ chính xác như bài báo
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Thêm đường dẫn
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.pso_svm_tuner import tune_svm_with_pso

# Cấu hình ghi nhật ký
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_breast_cancer_data(file_path, test_size=0.2, random_state=42):
    """Tải và chuẩn bị dữ liệu Breast Cancer."""
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Tệp không tìm thấy: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Đã tải dữ liệu: {df.shape}")
    
    # Tìm cột nhãn
    label_column = 'label'
    if label_column not in df.columns:
        raise ValueError("Không tìm thấy cột 'label'")
    
    # Trích xuất X, y
    y = df[label_column].values
    X = df.drop(columns=['label', 'id'])
    
    # Mã hóa nhãn nếu là chữ
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    
    X = X.values.astype(np.float32)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Chia dữ liệu: train={X_train.shape}, test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def main():
    """Chạy PSO tuning trên Breast Cancer dataset."""
    
    logger.info("\n" + "="*80)
    logger.info("PSO-SVM TUNING TRÊN BREAST CANCER DATASET")
    logger.info("="*80)
    logger.info("Mục tiêu: Đạt 99%+ độ chính xác (như bài báo)")
    logger.info("="*80 + "\n")
    
    try:
        # 1. Tải dữ liệu
        logger.info("[Bước 1] Tải dữ liệu...")
        X_train, X_test, y_train, y_test = load_and_prepare_breast_cancer_data(
            "../data/ung-thu-vu.csv"
        )
        
        # 2. Chạy PSO tuning
        logger.info("\n[Bước 2] Chạy PSO tuning để tìm siêu tham số tối ưu...")
        logger.info("(Điều này sẽ mất khoảng 2-5 phút)\n")
        
        best_svm, best_params, scaler = tune_svm_with_pso(
            X_train, y_train, X_test, y_test,
            n_particles=50,      # Nhiều particles = tìm tốt hơn
            n_iterations=100     # Nhiều iterations = hội tụ tốt hơn
        )
        
        # 3. Đánh giá trên training set
        logger.info("\n[Bước 3] Đánh giá mô hình...")
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        y_train_pred = best_svm.predict(X_train_scaled)
        y_test_pred = best_svm.predict(X_test_scaled)
        
        # Tính toán metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # 4. In kết quả
        logger.info("\n" + "="*80)
        logger.info("KẾT QUẢ PSO-SVM TUNING")
        logger.info("="*80)
        
        logger.info(f"\nSiêu Tham Số Tối Ưu:")
        logger.info(f"  C: {best_params['best_C']:.6f}")
        logger.info(f"  Gamma: {best_params['best_gamma']:.6f}")
        
        logger.info(f"\nChỉ Số Hiệu Suất:")
        logger.info(f"  Training Accuracy:   {train_accuracy:.6f} ({train_accuracy*100:.2f}%)")
        logger.info(f"  Test Accuracy:       {test_accuracy:.6f} ({test_accuracy*100:.2f}%)")
        logger.info(f"  Precision:           {test_precision:.6f}")
        logger.info(f"  Recall:              {test_recall:.6f}")
        logger.info(f"  F1-Score:            {test_f1:.6f}")
        
        logger.info(f"\nSo Sánh Với Kết Quả Trước:")
        logger.info(f"  SVM Tiêu Chuẩn:      96.49%")
        logger.info(f"  PSO-SVM Tuning:      {test_accuracy*100:.2f}%")
        logger.info(f"  Cải Thiện:           +{(test_accuracy-0.9649)*100:.2f}%")
        
        logger.info(f"\nSo Sánh Với Bài Báo:")
        logger.info(f"  Bài Báo (PSO-SVM):   99.6-100%")
        logger.info(f"  Kết Quả Chúng Ta:    {test_accuracy*100:.2f}%")
        
        if test_accuracy >= 0.99:
            logger.info(f"\n✓ THÀNH CÔNG! Đạt được 99%+ độ chính xác như bài báo!")
        elif test_accuracy >= 0.97:
            logger.info(f"\n✓ TỐT! Đạt được 97%+ độ chính xác (gần bài báo)")
        else:
            logger.info(f"\n⚠️  Còn cách bài báo. Cần tuning PSO tốt hơn.")
        
        logger.info(f"\n{'-'*80}")
        logger.info(f"Classification Report:")
        logger.info(f"{'-'*80}")
        logger.info(f"\n{classification_report(y_test, y_test_pred)}")
        
        logger.info("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
