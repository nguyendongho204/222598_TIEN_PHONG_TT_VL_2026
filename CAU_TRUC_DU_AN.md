# Cấu trúc dự án EBM + SVM Ensemble

## 📁 Thư mục gốc

| File / Thư mục | Chức năng |
|---|---|
| `training_demo.html` | **Giao diện web chính** — kéo-thả file CSV, chọn tham số, xem kết quả |
| `README.md` | Tổng quan dự án |
| `QUICKSTART_GUIDE.md` | Hướng dẫn chạy nhanh |
| `API_DOCUMENTATION.md` | Tài liệu API (bằng tiếng Anh) |
| `HOW_TO_TEST.md` | Hướng dẫn test dataset (tiếng Việt) |
| `HUONG_DAN_CAI_DAT.md` | Hướng dẫn cài đặt chi tiết (tiếng Việt) |
| `THONG_TIN_THU_VIEN_CONG_NGHE.md` | Thông tin thư viện và công nghệ (tiếng Việt) |
| `requirements.txt` | Danh sách thư viện Python cần cài |
| `Dockerfile` / `docker-compose.yml` | Cấu hình chạy bằng Docker |
| `runapp.txt` | Ghi chú chạy ứng dụng |

## 📁 `api_base/` — Backend (FastAPI)

| File / Thư mục | Chức năng |
|---|---|
| `run_api.py` | **Khởi động server** — chạy `python run_api.py` |
| `requirements.txt` | Thư viện Python cho backend |
| `start-all.bat` / `start.sh` | Script khởi động backend + frontend |
| `start-dev.ps1` / `start.ps1` | Script PowerShell khởi động dev |

### `api_base/app/` — Ứng dụng FastAPI

| File | Chức năng |
|---|---|
| `main.py` | Entry point, cấu hình FastAPI app, CORS, router |
| `config.py` | Cấu hình chung (host, port, ...) |
| `constants.py` | Hằng số (epochs, PCA threshold, ...) |
| `schemas.py` | Định nghĩa request/response (Pydantic models) |

#### `routers/` — API endpoints

| File | Chức năng |
|---|---|
| `train_endpoint.py` | **Endpoint chính** `POST /api/train-dataset` — upload CSV, train model |
| `base.py` | Endpoint cơ bản (health check, ...) |
| `file_upload.py` | Xử lý upload file |
| `ml_models.py` | Endpoint liên quan model |
| `execution_history.py` | Lịch sử chạy |
| `ensemble_api.py` | API cho ensemble model |
| `universal_pipeline.py` | Pipeline tổng quát |

#### `services/` — Business logic

| File | Chức năng |
|---|---|
| `training_service.py` | Logic chính: đọc CSV, xử lý dữ liệu, gọi model |

### `api_base/ml_models/` — Các mô hình ML

| File | Chức năng |
|---|---|---|
| `jepa_model.py` | **Mô hình JEPA** (PyTorch) — Joint Embedding Predictive Architecture, học tự giám sát |
| `jepa_svm_pipeline.py` | **Pipeline chính** — JEPA trích xuất embedding + SVM phân loại |

## 📁 `data/` — Dataset mẫu

| File | Mô tả |
|---|---|
| `Iris.csv` | Hoa Iris (3 lớp, 4 features) |
| `wine.csv` | Rượu vang (3 lớp, 13 features) |
| `breast-cancer.csv` | Ung thư vú (2 lớp) |
| `adult.csv` | Thu nhập người lớn (2 lớp) |
| `creditcard.csv` | Gian lận thẻ tín dụng |
| `creditcard_sample.csv` | Mẫu nhỏ của creditcard |
| `mnist_train.csv` / `mnist_test.csv` | Chữ số viết tay MNIST |
| `winequality-red.csv` | Chất lượng rượu vang đỏ |

## 📁 `tools/` — Công cụ phụ trợ

| File | Chức năng |
|---|---|
| `generate_test_datasets.py` | Tạo dataset test ngẫu nhiên |
| `generate_complex_datasets.py` | Tạo dataset phức tạp |
| `plot_compare.py` | Vẽ biểu đồ so sánh kết quả |
| `explain_demo.py` | Demo giải thích model |

## 📁 Khác

| Thư mục | Chức năng |
|---|---|
| `execution_history/` | Lịch sử các lần train đã chạy (JSON) |
| `models/` | Model đã train được lưu lại |
| `logs/` | File log |
| `utils/` | Tiện ích |
