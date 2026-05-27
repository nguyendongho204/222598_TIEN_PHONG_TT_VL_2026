"""
Công cụ So Sánh SVM Tiêu Chuẩn vs EBM-SVM Tối Ưu

Script này so sánh hiệu suất của SVM tiêu chuẩn với SVM được cải thiện bằng
kỹ thuật EBM (Energy-Based Model) trên các bộ dữ liệu phân loại.

Cách sử dụng:
    python so_sanh_svm_ebm_svm.py [--data-path ĐƯỜNG/DẪN] [--output ĐƯỜNG/DẪN/OUTPUT]
    
Ví dụ:
    python so_sanh_svm_ebm_svm.py --data-path data/ung-thu-vu.csv
    python so_sanh_svm_ebm_svm.py --data-path data/dữ-liệu.csv --output kết-quả/so-sánh.txt

Tác giả: Nhóm Phát Triển
Phiên bản: 1.0.0
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Thêm cha của thư mục vào đường dẫn
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_models.services.optimized_pipeline import OptimizedEBMSVMPipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cấu hình ghi nhật ký
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Baseline SVM Tiêu Chuẩn
# ============================================================================

class SVMChuanBaseline:
    """Huấn luyện SVM tiêu chuẩn với các tham số mặc định để so sánh."""
    
    @staticmethod
    def tải_chuẩn_bị_dữ_liệu(đường_dẫn_tệp, test_size=0.2, random_state=42):
        """Tải và chuẩn bị dữ liệu."""
        if not Path(đường_dẫn_tệp).exists():
            raise FileNotFoundError(f"Tệp không tìm thấy: {đường_dẫn_tệp}")
        
        df = pd.read_csv(đường_dẫn_tệp)
        logger.info(f"Đã tải dữ liệu: {df.shape}")
        
        # Tự động phát hiện cột nhãn
        cột_nhãn = None
        for tên_cột in ['label', 'target', 'class', 'y', 'species', 'income']:
            if tên_cột in df.columns:
                cột_nhãn = tên_cột
                break
        
        if cột_nhãn is None:
            raise ValueError("Không thể tìm thấy cột nhãn")
        
        # Trích xuất X, y
        y = df[cột_nhãn].values
        X = df.drop(columns=[cột_nhãn])
        
        # Mã hóa các tính năng phân loại
        for cột in X.select_dtypes(include=['object']).columns:
            bộ_mã_hóa = LabelEncoder()
            X[cột] = bộ_mã_hóa.fit_transform(X[cột])
        
        X = X.values.astype(np.float32)
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Chia tỉ lệ
        bộ_chia_tỉ_lệ = StandardScaler()
        X_train = bộ_chia_tỉ_lệ.fit_transform(X_train)
        X_test = bộ_chia_tỉ_lệ.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def huấn_luyện_và_đánh_giá(X_train, X_test, y_train, y_test):
        """Huấn luyện SVM tiêu chuẩn với các tham số mặc định."""
        logger.info("\nHuấn luyện SVM Tiêu Chuẩn (Các Tham Số Mặc Định)...")
        logger.info("  kernel: rbf")
        logger.info("  C: 1.0")
        logger.info("  gamma: scale")
        
        bắt_đầu = time.time()
        
        # Huấn luyện với các tham số mặc định
        máy_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        máy_svm.fit(X_train, y_train)
        
        # Dự đoán
        y_dự_đoán = máy_svm.predict(X_test)
        
        # Tính toán các chỉ số
        độ_chính_xác = accuracy_score(y_test, y_dự_đoán)
        độ_chính_xác_từng_lớp = precision_score(y_test, y_dự_đoán, average='weighted', zero_division=0)
        thu_hồi = recall_score(y_test, y_dự_đoán, average='weighted', zero_division=0)
        điểm_f1 = f1_score(y_test, y_dự_đoán, average='weighted', zero_division=0)
        
        thời_gian_đã_trôi = time.time() - bắt_đầu
        
        logger.info(f"  Thời gian huấn luyện: {thời_gian_đã_trôi:.2f}s")
        logger.info(f"  Độ chính xác: {độ_chính_xác:.4f} ({độ_chính_xác*100:.2f}%)")
        logger.info(f"  Độ chính xác từng lớp: {độ_chính_xác_từng_lớp:.4f}")
        logger.info(f"  Thu hồi: {thu_hồi:.4f}")
        logger.info(f"  Điểm F1: {điểm_f1:.4f}")
        
        return {
            "mô_hình": "SVM Tiêu Chuẩn",
            "độ_chính_xác": độ_chính_xác,
            "độ_chính_xác_từng_lớp": độ_chính_xác_từng_lớp,
            "thu_hồi": thu_hồi,
            "điểm_f1": điểm_f1,
            "thời_gian_huấn_luyện": thời_gian_đã_trôi
        }


# ============================================================================
# Hàm So Sánh
# ============================================================================

def so_sánh_các_mô_hình(đường_dẫn_tệp):
    """So sánh SVM tiêu chuẩn vs EBM-SVM tối ưu."""
    
    logger.info("\n" + "="*80)
    logger.info("SO SÁNH SVM TIÊU CHUẨN vs EBM-SVM TỐI ƯU")
    logger.info("="*80)
    logger.info(f"Bộ Dữ Liệu: {đường_dẫn_tệp}")
    logger.info(f"Dấu Thời Gian: {datetime.now().isoformat()}")
    
    try:
        # 1. Baseline SVM Tiêu Chuẩn
        logger.info("\n" + "-"*80)
        logger.info("1. BASELINE SVM TIÊU CHUẨN")
        logger.info("-"*80)
        
        X_train, X_test, y_train, y_test = SVMChuanBaseline.tải_chuẩn_bị_dữ_liệu(đường_dẫn_tệp)
        kết_quả_std = SVMChuanBaseline.huấn_luyện_và_đánh_giá(X_train, X_test, y_train, y_test)
        
        # 2. EBM-SVM Tối Ưu
        logger.info("\n" + "-"*80)
        logger.info("2. EBM-SVM TỐI ƯU")
        logger.info("-"*80)
        logger.info("Cấu Hình:")
        logger.info("  - Các epoch EBM: 2000 (so với tiêu chuẩn)")
        logger.info("  - Tốc độ học EBM: 0.0001")
        logger.info("  - Chiều sâu embedding EBM: 128")
        logger.info("  - Điều chỉnh lưới SVM: Bật")
        logger.info("  - Kết hợp đặc trưng: Original + Embeddings")
        logger.info("  - Ensemble: Bật")
        
        bắt_đầu = time.time()
        
        pipeline = OptimizedEBMSVMPipeline(
            use_tuning=True,
            ensemble_mode=True
        )
        
        kết_quả_ebm = pipeline.train(
            đường_dẫn_tệp,
            feature_selection="all"
        )
        
        thời_gian_đã_trôi = time.time() - bắt_đầu
        
        kết_quả_ebm_tóm_tắt = {
            "mô_hình": "EBM-SVM Tối Ưu",
            "độ_chính_xác": kết_quả_ebm["metadata"]["best_accuracy"],
            "độ_chính_xác_từng_lớp": kết_quả_ebm["metrics"]["enhanced"].get("precision", 0),
            "thu_hồi": kết_quả_ebm["metrics"]["enhanced"].get("recall", 0),
            "điểm_f1": kết_quả_ebm["metrics"]["enhanced"].get("f1", 0),
            "thời_gian_huấn_luyện": thời_gian_đã_trôi,
            "cải_thiện": kết_quả_ebm["metadata"]["improvement"],
            "cải_thiện_phần_trăm": kết_quả_ebm["metadata"]["improvement_pct"]
        }
        
        # 3. So Sánh Kết Quả
        logger.info("\n" + "="*80)
        logger.info("KẾT QUẢ SO SÁNH")
        logger.info("="*80)
        
        logger.info(f"\n{'Chỉ Số':<20} {'SVM Tiêu Chuẩn':<20} {'EBM-SVM':<20} {'Sự Khác Biệt':<15}")
        logger.info("-" * 75)
        
        các_chỉ_số = ['độ_chính_xác', 'độ_chính_xác_từng_lớp', 'thu_hồi', 'điểm_f1']
        for chỉ_số in các_chỉ_số:
            giá_trị_std = kết_quả_std[chỉ_số]
            giá_trị_ebm = kết_quả_ebm_tóm_tắt[chỉ_số]
            khác_biệt = giá_trị_ebm - giá_trị_std
            
            logger.info(f"{chỉ_số.replace('_', ' ').title():<20} {giá_trị_std:<20.4f} {giá_trị_ebm:<20.4f} {khác_biệt:+.4f}")
        
        logger.info(f"{'Thời gian huấn luyện':<20} {kết_quả_std['thời_gian_huấn_luyện']:<20.2f}s {kết_quả_ebm_tóm_tắt['thời_gian_huấn_luyện']:<20.2f}s {kết_quả_ebm_tóm_tắt['thời_gian_huấn_luyện']-kết_quả_std['thời_gian_huấn_luyện']:+.2f}s")
        
        # 4. Tóm Tắt
        logger.info("\n" + "="*80)
        logger.info("TÓM TẮT")
        logger.info("="*80)
        
        cải_thiện = kết_quả_ebm_tóm_tắt['độ_chính_xác'] - kết_quả_std['độ_chính_xác']
        cải_thiện_phần_trăm = (cải_thiện / kết_quả_std['độ_chính_xác']) * 100 if kết_quả_std['độ_chính_xác'] > 0 else 0
        
        logger.info(f"\nĐộ Chính Xác SVM Tiêu Chuẩn:     {kết_quả_std['độ_chính_xác']:.4f} ({kết_quả_std['độ_chính_xác']*100:.2f}%)")
        logger.info(f"Độ Chính Xác EBM-SVM Tối Ưu:   {kết_quả_ebm_tóm_tắt['độ_chính_xác']:.4f} ({kết_quả_ebm_tóm_tắt['độ_chính_xác']*100:.2f}%)")
        logger.info(f"\nCải Thiện:                      +{cải_thiện:.4f} (+{cải_thiện_phần_trăm:.2f}%)")
        
        if cải_thiện > 0:
            logger.info(f"\n✓ THÀNH CÔNG: EBM-SVM vượt trội hơn SVM Tiêu Chuẩn {cải_thiện_phần_trăm:.2f}%")
        else:
            logger.info(f"\n✗ EBM-SVM không cải thiện hơn SVM Tiêu Chuẩn")
        
        logger.info(f"\nSo Sánh Thời Gian Huấn Luyện:")
        logger.info(f"  SVM Tiêu Chuẩn:   {kết_quả_std['thời_gian_huấn_luyện']:.2f}s")
        logger.info(f"  EBM-SVM:          {kết_quả_ebm_tóm_tắt['thời_gian_huấn_luyện']:.2f}s")
        logger.info(f"  Tỷ Lệ:            {kết_quả_ebm_tóm_tắt['thời_gian_huấn_luyện']/kết_quả_std['thời_gian_huấn_luyện']:.1f}x")
        
        # 5. Lưu Kết Quả
        tệp_kết_quả = f"kết_quả/so_sánh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        Path(tệp_kết_quả).parent.mkdir(parents=True, exist_ok=True)
        
        with open(tệp_kết_quả, 'w', encoding='utf-8') as f:
            f.write("SO SÁNH SVM TIÊU CHUẨN vs EBM-SVM TỐI ƯU\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Bộ Dữ Liệu: {đường_dẫn_tệp}\n")
            f.write(f"Dấu Thời Gian: {datetime.now().isoformat()}\n\n")
            
            f.write("KẾT QUẢ:\n")
            f.write(f"{'Chỉ Số':<20} {'SVM Tiêu Chuẩn':<20} {'EBM-SVM':<20} {'Sự Khác Biệt':<15}\n")
            f.write("-" * 75 + "\n")
            
            for chỉ_số in các_chỉ_số:
                giá_trị_std = kết_quả_std[chỉ_số]
                giá_trị_ebm = kết_quả_ebm_tóm_tắt[chỉ_số]
                khác_biệt = giá_trị_ebm - giá_trị_std
                f.write(f"{chỉ_số.replace('_', ' ').title():<20} {giá_trị_std:<20.4f} {giá_trị_ebm:<20.4f} {khác_biệt:+.4f}\n")
            
            f.write("\n\nTÓM TẮT:\n")
            f.write(f"Độ Chính Xác SVM Tiêu Chuẩn:      {kết_quả_std['độ_chính_xác']:.4f}\n")
            f.write(f"Độ Chính Xác EBM-SVM Tối Ưu:    {kết_quả_ebm_tóm_tắt['độ_chính_xác']:.4f}\n")
            f.write(f"Cải Thiện:                       +{cải_thiện:.4f} (+{cải_thiện_phần_trăm:.2f}%)\n")
        
        logger.info(f"\nKết Quả được lưu tại: {tệp_kết_quả}")
        
        return kết_quả_std, kết_quả_ebm_tóm_tắt
        
    except Exception as e:
        logger.error(f"So sánh thất bại: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# Chương Trình Chính
# ============================================================================

def main():
    """Điểm vào chính."""
    parser = argparse.ArgumentParser(
        description="So sánh SVM Tiêu Chuẩn vs EBM-SVM Tối Ưu"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Đường dẫn đến dữ liệu huấn luyện (tệp CSV)"
    )
    parser.add_argument(
        "--output",
        default="kết_quả/so_sánh.txt",
        help="Tệp đầu ra cho kết quả"
    )
    
    args = parser.parse_args()
    
    try:
        kết_quả_std, kết_quả_ebm = so_sánh_các_mô_hình(args.data_path)
        return 0 if kết_quả_std is not None else 1
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
