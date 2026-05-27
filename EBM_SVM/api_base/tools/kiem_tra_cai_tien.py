"""
Kiểm Tra Xem EBM-SVM Có Thực Sự Cải Thiện Không

Script này chạy so sánh trực tiếp SVM tiêu chuẩn vs EBM-SVM với cấu hình đã sửa.
"""

import sys
import logging
from pathlib import Path

# Thêm đường dẫn
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.so_sanh_svm_ebm_svm import so_sánh_các_mô_hình

# Cấu hình ghi nhật ký
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Chạy so sánh với cấu hình đã sửa."""
    
    logger.info("\n" + "="*80)
    logger.info("KIỂM TRA CẢI TIẾN EBM-SVM (Phiên Bản Đã Sửa)")
    logger.info("="*80)
    logger.info("\nCấu Hình Mới (Ngắn Hạn Overfitting):")
    logger.info("  - Embedding Dim: 64 (giảm từ 128)")
    logger.info("  - Hidden Dim: 256 (giảm từ 512)")
    logger.info("  - Epochs: 800 (giảm từ 2000)")
    logger.info("  - Learning Rate: 0.001 (tăng từ 0.0001)")
    logger.info("  - Top Embeddings: 32 (để tránh nhiễu)")
    logger.info("="*80 + "\n")
    
    # So sánh trên dữ liệu ung thư vú
    logger.info("Chạy so sánh trên bộ dữ liệu Ung Thư Vú (Breast Cancer)...")
    kết_quả_std, kết_quả_ebm = so_sánh_các_mô_hình("../data/ung-thu-vu.csv")
    
    if kết_quả_std and kết_quả_ebm:
        # Tính toán cải thiện
        cải_thiện = kết_quả_ebm['độ_chính_xác'] - kết_quả_std['độ_chính_xác']
        cải_thiện_phần_trăm = (cải_thiện / kết_quả_std['độ_chính_xác']) * 100
        
        logger.info("\n" + "="*80)
        logger.info("KẾT QUẢ KIỂM TRA")
        logger.info("="*80)
        
        if cải_thiện > 0:
            logger.info(f"\n✓ THÀNH CÔNG!")
            logger.info(f"  EBM-SVM cải thiện được {cải_thiện_phần_trăm:.2f}%")
            logger.info(f"  SVM: {kết_quả_std['độ_chính_xác']:.4f} ({kết_quả_std['độ_chính_xác']*100:.2f}%)")
            logger.info(f"  EBM-SVM: {kết_quả_ebm['độ_chính_xác']:.4f} ({kết_quả_ebm['độ_chính_xác']*100:.2f}%)")
            logger.info(f"  Cải thiện: +{cải_thiện:.4f} (+{cải_thiện_phần_trăm:.2f}%)")
        else:
            logger.info(f"\n✗ CHƯA CẢI THIỆN")
            logger.info(f"  EBM-SVM vẫn thấp hơn SVM {abs(cải_thiện_phần_trăm):.2f}%")
            logger.info(f"  SVM: {kết_quả_std['độ_chính_xác']:.4f} ({kết_quả_std['độ_chính_xác']*100:.2f}%)")
            logger.info(f"  EBM-SVM: {kết_quả_ebm['độ_chính_xác']:.4f} ({kết_quả_ebm['độ_chính_xác']*100:.2f}%)")
            logger.info(f"  Kém hơn: {cải_thiện:.4f} ({cải_thiện_phần_trăm:.2f}%)")
            logger.info("\n⚠️  EBM không giúp được trên bộ dữ liệu này.")
            logger.info("   Có thể dữ liệu là tuyến tính hoặc EBM cần tuning hơn nữa.")
        
        logger.info("="*80)
        return 0 if cải_thiện > 0 else 1
    else:
        logger.error("So sánh thất bại")
        return 1


if __name__ == "__main__":
    sys.exit(main())
