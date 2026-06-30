# 🚀 Hướng Dẫn Mở Web và Test Dataset

## 📋 Yêu Cầu Trước Khi Chạy
- Python 3.11+ đã cài
- Chạy: `pip install -r requirements.txt`
- Chạy: `pip install -r api_base/requirements.txt`

---

## ✅ Bước 1: Khởi Động API Server

### Trên Windows (PowerShell)
```powershell
cd D:\Nam4\ThucTap\EBM_SVM\api_base
python run_api.py
```

### Trên Linux/Mac
```bash
cd api_base
python run_api.py
```

**Kết quả khi khởi động thành công:**
```
INFO:     Started server process [xxxx]
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

⏳ **Chờ cho đến khi thấy dòng trên**, API đã sẵn sàng!

---

## ✅ Bước 2: Mở Web Interface

### Cách 1: Mở Trực Tiếp File
**Trên Windows:**
```
Nhấn Windows + R → gõ:
file:///D:/Nam4/ThucTap/EBM_SVM/training_demo.html
```

**Trên Linux/Mac:**
```
file:///home/user/Nam4/ThucTap/EBM_SVM/training_demo.html
```

### Cách 2: Mở Bằng VS Code
1. Mở folder `EBM_SVM` trong VS Code
2. Nhấp chuột phải trên `training_demo.html` → **Open with Live Server**
3. Hoặc nhấn Alt+L (nếu có Live Server extension)

### Cách 3: Drag-Drop
- Tìm file `training_demo.html` trong File Explorer
- Drag nó vào trình duyệt (Chrome, Firefox, Edge)

---

## ✅ Bước 3: Test Dataset

### Giao Diện Web
```
┌─────────────────────────────────────────┐
│  ENERGY-BASED MODEL (EBM) SVM           │
│  Xây Dựng EBM Kết Hợp SVM               │
│                                         │
│  📤 Kéo thả file CSV vào đây            │
│                                         │
│  ⚙️ Cấu Hình (Config):                  │
│  - Test Size: 20%                       │
│  - EBM Epochs: 100                      │
│  - PCA Variance: 95%                    │
│  - EBM Weight: 0.5 (deprecated)        │
│                                         │
│  🚀 [Chạy Phân Tích]                    │
│                                         │
│  📊 Kết Quả (SVM vs EBM+SVM)            │
└─────────────────────────────────────────┘
```

### Các Dataset Có Sẵn

Tất cả dataset có sẵn trong folder `data/`:

| File | Mô Tả | Dòng | Tính Năng |
|------|-------|------|----------|
| `Iris.csv` | Phân loại hoa Iris | 150 | 4 features |
| `breast-cancer.csv` | Ung thư vú | 683 | 10 features |
| `wine.csv` | Phân loại rượu | 178 | 13 features |
| `winequality-red.csv` | Chất lượng rượu | 1599 | 11 features |
| `adult.csv` | Dự đoán thu nhập | 32561 | 14 features |
| `creditcard.csv` | Gian lận thẻ tín dụng | 284807 | 30 features |
| `mnist_train.csv` | Chữ số viết tay | 60000 | 784 features |
| `mnist_test.csv` | Chữ số viết tay (test) | 10000 | 784 features |

---

## 🧪 Quy Trình Test

### 1️⃣ Upload Dataset
```
Cách A: Kéo thả file CSV vào vùng upload
Cách B: Nhấp vào vùng upload → chọn file → chọn từ data/ → OK
```

### 2️⃣ Điều Chỉnh Config (Tùy Chọn)
```
- Test Size: 5% ↔ 50% (dữ liệu kiểm thử)
- EBM Epochs: 10 ↔ 500 (vòng lặp huấn luyện)
- PCA Variance: 50% ↔ 99% (giảm chiều)
- EBM Weight: 0% ↔ 100% (deprecated — giữ lại để tương thích)
```

### 3️⃣ Nhấn [Chạy Phân Tích]
- Tiến độ hiển thị trong progress bar
- API xử lý: load, preprocess, train models
- Chờ ~10-30 giây (tùy kích thước dataset)

### 4️⃣ Xem Kết Quả

**📊 Bảng So Sánh:**
```
┌─────────────────┬──────────┬──────────┬──────────────────┐
│ Model           │ SVM      │ EBM      │ Ensemble         │
│                 │ (baseline)│         │ (Feature Aug.)   │
├─────────────────┼──────────┼──────────┼──────────────────┤
│ Accuracy (%)    │ 85.0%    │ 83.0%    │ 85.6%            │
│ Improvement     │ Baseline │ -2.4%    │ +0.7% ✓          │
└─────────────────┴──────────┴──────────┴──────────────────┘
```

**📈 Biểu Đồ Tương Tác:**
- So sánh 4 model (SVM, EBM, Ensemble, Final Best)
- Hover để xem giá trị chính xác
- Chart.js visualization

---

## 🐛 Troubleshooting

### ❌ Lỗi: "API không phản hồi"
```
→ Kiểm tra API server chạy trên http://127.0.0.1:8000
→ Chạy lại: cd api_base && python run_api.py
→ Kiểm tra port 8000 có bị chiếm không
```

### ❌ Lỗi: "File không hợp lệ"
```
→ Kiểm tra file CSV có header không
→ Dòng đầu phải là tên cột
→ Dữ liệu phải có ít nhất 2 dòng
```

### ❌ Lỗi: "Dataset quá lớn"
```
→ Giới hạn: tối đa 100MB
→ Nếu dataset > 100MB, lấy sample nhỏ hơn
→ Ví dụ: Lấy 10,000 dòng đầu từ creditcard.csv
```

### ❌ Lỗi: "Browser không load file html"
```
→ Kiểm tra đường dẫn:
   file:///D:/Nam4/ThucTap/EBM_SVM/training_demo.html
→ Không dùng backslash (\), chỉ dùng forward slash (/)
→ Thử dùng Live Server thay vì file:// URI
```

---

## 🎯 Test Nhanh (Quick Test)

### Bài 1: Test Iris Dataset (30 giây)
```
1. Khởi động API: cd api_base && python run_api.py
2. Mở web: file:///D:/Nam4/ThucTap/EBM_SVM/training_demo.html
3. Kéo thả: data/Iris.csv
4. Bấm: [Chạy Phân Tích]
5. Xem kết quả → Improvement % phải > 0%
```

### Bài 2: Test Breast Cancer (1 phút)
```
1. Kéo thả: data/breast-cancer.csv
2. Config: Test Size = 30%, EBM Epochs = 200
3. Bấm: [Chạy Phân Tích]
4. Kết quả nên có accuracy > 95%
```

### Bài 3: Test Custom Dataset
```
1. Chuẩn bị file CSV của riêng bạn
2. Format: Hàng 1 = Header, Các hàng sau = Dữ liệu
3. Kéo thả vào web
4. Đợi kết quả
```

---

## 📝 Ghi Chú Về Kết Quả

- **SVM Accuracy**: Baseline (mô hình cơ bản)
- **EBM Accuracy**: Mô hình EBM độc lập
- **Ensemble Accuracy**: SVM với Feature Augmentation (PCA + EBM embeddings)
- **Improvement %**: `(Best - SVM) / SVM * 100%` (best = max(svm, ebm, ensemble))
  - ✅ Improvement > 0% = Chứng minh luận văn thành công
  - ❌ Improvement < 0% = EBM chưa tối ưu

---

## 🚀 Chế Độ Dev (Development Mode)

### Chạy API với Auto-Reload
```powershell
cd api_base
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Xem Logs Chi Tiết
API logs được lưu tại: `api_base/api.log`
```powershell
Get-Content api_base/api.log -Tail 50  # Xem 50 dòng cuối
```

### API Documentation
Mỗi khi API chạy, có sẵn:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra API server đang chạy
2. Xem [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
3. Xem [HUONG_DAN_CAI_DAT.md](HUONG_DAN_CAI_DAT.md)
4. Kiểm tra logs trong `api_base/api.log`

---

**🎉 Chúc bạn test thành công! Happy Coding! 🚀**
