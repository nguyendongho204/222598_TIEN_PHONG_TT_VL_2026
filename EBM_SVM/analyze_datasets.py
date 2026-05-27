"""
Phân Tích & Xác Định Dữ Liệu Thực vs Giả Tạo

Kiểm tra từng file để biết:
1. Dữ liệu có bao nhiêu rows/columns?
2. Tên file có gợi ý gì? (make_classification → giả tạo)
3. Thống kê (mean, std, min, max) có bất thường không?
4. Dữ liệu từ đâu?
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Phân tích & xác định nguồn gốc dữ liệu."""
    
    # Cơ sở dữ liệu nổi tiếng (thực)
    KNOWN_REAL_DATASETS = {
        'breast-cancer': 'UCI ML - Breast Cancer Wisconsin',
        'iris': 'UCI ML - Iris',
        'wine': 'UCI ML - Wine',
        'winequality': 'UCI ML - Wine Quality',
        'creditcard': 'Kaggle - Credit Card Fraud',
        'mnist': 'Kaggle - MNIST Handwritten Digits'
    }
    
    # Tên file gợi ý dữ liệu giả tạo
    SYNTHETIC_INDICATORS = [
        'make_classification',  # scikit-learn synthetic
        'make_moons',           # scikit-learn synthetic
        'make_circles',         # scikit-learn synthetic
        'circles',              # synthetic
        'moons',                # synthetic
        'generated',            # synthetic
        'synthetic',            # synthetic
        'test_data',            # test data (có thể giả tạo)
        'train_data',           # train data (có thể giả tạo)
        'uploaded_data'         # user-uploaded (không rõ)
    ]
    
    @staticmethod
    def analyze_file(file_path):
        """Phân tích một file dữ liệu."""
        
        file_name = Path(file_path).stem
        file_size = Path(file_path).stat().st_size / 1024  # KB
        
        try:
            df = pd.read_csv(file_path)
            
            # Cơ bản
            n_rows = len(df)
            n_cols = len(df.columns)
            
            # Xác định loại dữ liệu
            data_type = "KHÔNG RÕ"
            is_synthetic = False
            
            # Kiểm tra tên file
            for synthetic_name in DataAnalyzer.SYNTHETIC_INDICATORS:
                if synthetic_name.lower() in file_name.lower():
                    data_type = "GIẢI TẠO (Synthetic)"
                    is_synthetic = True
                    break
            
            if not is_synthetic:
                for real_name, source in DataAnalyzer.KNOWN_REAL_DATASETS.items():
                    if real_name.lower() in file_name.lower():
                        data_type = f"THỰC (Real) - {source}"
                        break
            
            # Nếu vẫn không biết
            if data_type == "KHÔNG RÕ":
                if n_rows > 5000:
                    data_type = "THỰC (Likely) - Kích thước lớn"
                elif n_rows < 1000 and 'test' not in file_name.lower():
                    data_type = "GIẢI TẠO (Likely) - Kích thước nhỏ"
                else:
                    data_type = "CẦN KIỂM TRA THÊM"
            
            # Thống kê
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            stats = {
                'n_rows': n_rows,
                'n_cols': n_cols,
                'file_size_kb': file_size,
                'numeric_cols': len(numeric_cols),
                'data_type': data_type,
                'is_synthetic': is_synthetic
            }
            
            # Thống kê chi tiết cho numeric columns
            if len(numeric_cols) > 0:
                stats['mean_values'] = df[numeric_cols].mean().values
                stats['std_values'] = df[numeric_cols].std().values
                stats['min_values'] = df[numeric_cols].min().values
                stats['max_values'] = df[numeric_cols].max().values
            
            return stats, df
            
        except Exception as e:
            logger.error(f"Lỗi đọc {file_name}: {str(e)}")
            return None, None


def main():
    """Phân tích tất cả file dữ liệu."""
    
    logger.info("\n" + "="*100)
    logger.info("PHÂN TÍCH DỮ LIỆU - THỰC vs GIẢI TẠO")
    logger.info("="*100)
    
    data_dir = Path(".")
    csv_files = sorted(data_dir.glob("*.csv"))
    
    analyzer = DataAnalyzer()
    results = []
    
    logger.info(f"\nTìm thấy {len(csv_files)} file CSV\n")
    
    for file_path in csv_files:
        stats, df = analyzer.analyze_file(file_path)
        
        if stats:
            results.append((file_path.name, stats))
            
            logger.info(f"{'='*100}")
            logger.info(f"📄 {file_path.name}")
            logger.info(f"{'-'*100}")
            
            logger.info(f"Kích thước:        {stats['n_rows']:,} rows × {stats['n_cols']} columns ({stats['file_size_kb']:.1f} KB)")
            logger.info(f"Loại dữ liệu:     {stats['data_type']}")
            logger.info(f"Giải tạo?:        {'✓ CÓ' if stats['is_synthetic'] else '✗ KHÔNG'}")
            logger.info(f"Numeric columns:  {stats['numeric_cols']}")
            
            # Đặc điểm gợi ý
            hints = []
            
            if stats['n_rows'] < 500:
                hints.append("🔍 Kích thước nhỏ → Có thể giải tạo hoặc sample")
            elif stats['n_rows'] > 100000:
                hints.append("🔍 Kích thước lớn → Có thể thực hoặc mock data")
            
            if 'mean_values' in stats:
                means = stats['mean_values']
                stds = stats['std_values']
                
                # Kiểm tra nếu all features có phân phối chuẩn giống nhau
                mean_of_means = np.mean(means)
                std_of_stds = np.std(stds)
                
                if std_of_stds < 0.1 * mean_of_means:
                    hints.append("🔍 Phân phối rất đều → Gợi ý giải tạo")
                
                # Kiểm tra nếu values quá round
                if all(str(v).endswith('0') for v in means if v > 1):
                    hints.append("🔍 Values quá round → Gợi ý giải tạo")
            
            if hints:
                logger.info(f"Dấu hiệu:")
                for hint in hints:
                    logger.info(f"  {hint}")
    
    # Bảng tóm tắt
    logger.info("\n" + "="*100)
    logger.info("BẢNG TÓM TẮT")
    logger.info("="*100 + "\n")
    
    logger.info(f"{'File Name':<30} | {'Loại':<15} | {'Rows':<10} | {'Cols':<6} | {'Synthetic?':<12}")
    logger.info("-" * 100)
    
    real_count = 0
    synthetic_count = 0
    unknown_count = 0
    
    for file_name, stats in results:
        loai = stats['data_type'].split('-')[0].strip()  # Lấy phần đầu
        is_syn = "✓ CÓ" if stats['is_synthetic'] else "✗ KHÔNG"
        
        logger.info(f"{file_name:<30} | {loai:<15} | {stats['n_rows']:<10,} | {stats['n_cols']:<6} | {is_syn:<12}")
        
        if stats['is_synthetic']:
            synthetic_count += 1
        elif "THỰC" in stats['data_type']:
            real_count += 1
        else:
            unknown_count += 1
    
    logger.info("-" * 100)
    logger.info(f"\nThống kê:")
    logger.info(f"  ✓ Dữ liệu THỰC:     {real_count}")
    logger.info(f"  ✓ Dữ liệu GIẢI TẠO: {synthetic_count}")
    logger.info(f"  ? CẦN KIỂM TRA:      {unknown_count}")
    logger.info(f"  TỔNG:               {len(results)}")
    
    # Khuyến nghị
    logger.info("\n" + "="*100)
    logger.info("KHUYẾN NGHỊ")
    logger.info("="*100)
    logger.info("\n✓ DÙNG NHỮNG DỮ LIỆU NÀY ĐỂ TEST EBM-SVM:")
    logger.info("  - Breast Cancer (THỰC, 569 rows)")
    logger.info("  - Wine (THỰC, 178 rows)")
    logger.info("  - Iris (THỰC, 150 rows)")
    logger.info("  - Wine Quality (THỰC, 1599 rows)")
    logger.info("  - Credit Card (THỰC, 284,807 rows)")
    
    logger.info("\n✗ KHÔNG DÙNG NHỮNG DỮ LIỆU NÀY:")
    logger.info("  - Circles (GIẢI TẠO - scikit-learn)")
    logger.info("  - Moons (GIẢI TẠO - scikit-learn)")
    logger.info("  - Classification Complex (GIẢI TẠO - scikit-learn)")
    logger.info("  - Make Classification (GIẢI TẠO - scikit-learn)")
    logger.info("  (Vì không phải dữ liệu thực từ nghiên cứu)")
    
    logger.info("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
