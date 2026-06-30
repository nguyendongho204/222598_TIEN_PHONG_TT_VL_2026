# Thông tin thư viện và công nghệ sử dụng

## 1. Ngôn ngữ lập trình

### Python 3.11
- **Mục đích**: Ngôn ngữ chính cho toàn bộ backend, xử lý dữ liệu, huấn luyện mô hình
- **Lý do chọn**: Hệ sinh thái thư viện ML/DL phong phú (PyTorch, scikit-learn), cú pháp đơn giản

### JavaScript (ES6+)
- **Mục đích**: Xây dựng giao diện người dùng React
- **Lý do chọn**: Tương tác với API backend qua fetch, hiển thị biểu đồ kết quả

---

## 2. Thư viện Machine Learning / Deep Learning

### PyTorch 2.0
- **Mục đích**: Xây dựng và huấn luyện Energy-Based Model (EBM)
- **Chức năng trong đồ án**: 
  - Xây dựng mạng nơ-ron 4 lớp [256→128→64→32] cho EBM
  - Học hàm năng lượng E(x, y) qua cơ chế lan truyền ngược
  - Tối ưu bằng thuật toán Adam với ReduceLROnPlateau

### scikit-learn 1.3
- **Mục đích**: Triển khai SVM, PCA, tiền xử lý dữ liệu
- **Chức năng trong đồ án**:
  - `SVC(kernel='rbf')` — SVM với kernel RBF
  - `PCA(n_components=0.99)` — Giảm chiều dữ liệu
  - `StandardScaler` — Chuẩn hóa dữ liệu
  - `OneHotEncoder` — Mã hóa đặc trưng phân loại
  - `train_test_split` — Chia tập dữ liệu (stratified)
  - `accuracy_score`, `classification_report`, `confusion_matrix` — Đánh giá

### NumPy 1.24
- **Mục đích**: Tính toán ma trận, xử lý mảng đa chiều
- **Chức năng**: Xử lý dữ liệu đầu vào, chuyển đổi giữa numpy array và tensor

### Pandas 2.0
- **Mục đích**: Đọc và xử lý dữ liệu dạng bảng (CSV)
- **Chức năng**: Đọc file CSV, phát hiện cột categorical, xử lý dữ liệu thiếu

---

## 3. Framework Web

### FastAPI 0.104
- **Mục đích**: Xây dựng REST API backend
- **Chức năng trong đồ án**:
  - Endpoint `POST /api/train-dataset` — Huấn luyện mô hình từ CSV
  - Endpoint `POST /api/ensemble/predict` — Dự đoán với model đã train
  - Swagger UI tự động tại `/docs`
  - CORS middleware cho phép frontend gọi API

### Uvicorn 0.24
- **Mục đích**: ASGI server chạy FastAPI
- **Chức năng**: Chạy ứng dụng FastAPI, hỗ trợ async/await

### React 18 + Vite
- **Mục đích**: Xây dựng giao diện người dùng
- **Chức năng trong đồ án**:
  - Giao diện upload CSV kéo-thả
  - Form cấu hình tham số (test_size, epochs, PCA, weight)
  - Hiển thị biểu đồ so sánh accuracy bằng Recharts

---

## 4. Thuật toán

### Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Tham số**: C được tối ưu bằng grid search trên [1, 10, 50, 100]
- **Vai trò**: Mô hình baseline, thành phần chính trong ensemble

### Energy-Based Model (EBM)
- **Kiến trúc**: Mạng nơ-ron 4 lớp ẩn [256→128→64→32], đầu vào là (features + one-hot class), đầu ra là năng lượng
- **Loss function**: Negative log-softmax của energy
- **Optimizer**: Adam + ReduceLROnPlateau (giảm LR khi loss plateau)
- **Early stopping**: Dừng nếu loss không cải thiện sau 50 epochs
- **Vai trò**: Cung cấp đặc trưng bổ sung cho SVM (Feature Augmentation)

### PCA (Principal Component Analysis)
- **Ngưỡng**: Giữ 99% phương sai
- **Vai trò**: Giảm chiều dữ liệu, loại bỏ nhiễu, tăng tốc huấn luyện

### Ensemble — Feature Augmentation
- **Chiến lược**: Kết hợp EBM và SVM bằng cách dùng confidence scores của EBM làm đặc trưng đầu vào cho SVM, thay vì weighted voting
- **Lý do**: Cho phép SVM học từ các patterns mà EBM phát hiện được

---

## 5. Công cụ phát triển

| Công cụ | Phiên bản | Mục đích |
|---------|-----------|----------|
| Git | 2.x | Quản lý phiên bản mã nguồn |
| VS Code | Latest | Soạn thảo code, debug |
| Python venv | — | Môi trường ảo Python, cách ly thư viện |
| npm | 9+ | Quản lý packages cho React frontend |
| Docker | — | Đóng gói và triển khai ứng dụng |

---

## 6. Hệ điều hành và môi trường triển khai

| Môi trường | Mô tả |
|-----------|-------|
| Windows 10/11 | Môi trường phát triển chính |
| Linux (Docker) | Môi trường triển khai (tùy chọn) |

---

## 7. Danh sách package Python (requirements.txt)

```
fastapi>=0.104.0          # Web framework
uvicorn[standard]>=0.24.0  # ASGI server
pandas>=2.0.0              # Xử lý dữ liệu
numpy>=1.24.0              # Tính toán số học
scikit-learn>=1.3.0        # ML (SVM, PCA, preprocessing)
torch>=2.0.0               # Deep learning (EBM)
python-multipart            # Upload file
python-dotenv               # Cấu hình môi trường
tqdm                        # Progress bar
pydantic                    # Validate dữ liệu API
```
