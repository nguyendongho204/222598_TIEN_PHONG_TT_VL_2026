# Hướng dẫn cài đặt và chạy ứng dụng EBM + SVM Ensemble

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|-----------|---------|
| Hệ điều hành | Windows 10/11, Linux, macOS |
| Python | 3.10 hoặc 3.11 |
| RAM | Tối thiểu 4GB |
| Ổ cứng | Tối thiểu 2GB trống |
| Node.js | 18.x trở lên (nếu chạy React frontend) |

## 2. Cài đặt Python

Tải Python 3.11 từ: https://www.python.org/downloads/

Khi cài đặt, **nhớ tick** chọn **"Add Python to PATH"**.

Kiểm tra sau khi cài:
```powershell
python --version
pip --version
```

## 3. Tạo môi trường ảo (venv)

Mở PowerShell tại thư mục dự án:
```powershell
cd D:\Nam4\ThucTap\EBM_SVM
python -m venv venv
```

Kích hoạt môi trường ảo:
```powershell
.\venv\Scripts\activate
```

Sau khi kích hoạt, thấy dòng `(venv)` xuất hiện trước đường dẫn.

## 4. Cài đặt thư viện Python

Cài đặt tất cả thư viện cần thiết:
```powershell
pip install -r requirements.txt
pip install -r api_base\requirements.txt
```

Hoặc cài riêng từng gói chính:
```powershell
pip install fastapi uvicorn pandas numpy scikit-learn torch torchvision
pip install python-multipart python-dotenv tqdm pydantic
```

## 5. Kiểm tra thư viện đã cài đúng

```powershell
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
```

## 6. Cài đặt Node.js (cho React Frontend)

Tải Node.js 18+ từ: https://nodejs.org/

Kiểm tra:
```powershell
node --version
npm --version
```

Cài đặt dependencies cho frontend:
```powershell
cd api_base\frontend
npm install
```

## 7. Chạy ứng dụng

### Cách 1: Chạy Backend (API Server)

**Terminal 1** - Backend FastAPI:
```powershell
cd D:\Nam4\ThucTap\EBM_SVM\api_base
python run_api.py
```

Sau khi chạy thành công, truy cập:
- Giao diện API: http://localhost:8000
- Swagger UI (tài liệu API): http://localhost:8000/docs

### Cách 2: Chạy Frontend React (tùy chọn)

**Terminal 2** - Frontend React:
```powershell
cd D:\Nam4\ThucTap\EBM_SVM\api_base\frontend
npm run dev
```

Sau đó truy cập: http://localhost:3000

### Cách 3: Dùng file HTML tĩnh (đơn giản nhất)

Mở file sau trong trình duyệt:
```
D:\Nam4\ThucTap\EBM_SVM\training_demo.html
```

## 8. Kiểm tra hoạt động

### Kiểm tra API health:
Mở trình duyệt và truy cập: http://localhost:8000/api/ensemble/health

Kết quả mong đợi:
```json
{
  "status": "healthy",
  "model_type": "EBM + SVM Ensemble V3",
  "model_trained": false,
  "timestamp": "..."
}
```

### Kiểm tra train với dataset mẫu (Iris):
```powershell
cd D:\Nam4\ThucTap\EBM_SVM\api_base
python -c "
import requests
files = {'file': open('../data/Iris.csv', 'rb')}
params = {'test_size': 0.2, 'ebm_epochs': 50, 'pca_variance': 0.95}
r = requests.post('http://localhost:8000/api/train-dataset', files=files, data=params)
print(r.json())
"
```

## 9. Cấu trúc thư mục chính

```
EBM_SVM/
├── api_base/                    # Backend FastAPI
│   ├── app/
│   │   ├── main.py             # Khởi tạo FastAPI
│   │   ├── constants.py        # Hằng số cấu hình
│   │   ├── routers/
│   │   │   └── train_endpoint.py  # API train chính
│   │   └── services/
│   │       └── training_service.py  # Xử lý nghiệp vụ
│   ├── ml_models/
│   │   ├── energy_based_model.py    # EBM (PyTorch)
│   │   └── ebm_svm_ensemble_v3.py   # Ensemble chính
│   ├── frontend/                # React UI
│   └── run_api.py               # Script chạy backend
├── data/                        # Dataset mẫu
├── training_demo.html            # Giao diện web demo
└── requirements.txt              # Thư viện Python
```

## 10. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
|-----|-------------|----------------|
| `'python' not recognized` | Python chưa có trong PATH | Cài lại Python, tick "Add to PATH" |
| `No module named 'torch'` | Chưa cài PyTorch | `pip install torch` |
| `Port 8000 already in use` | Cổng bị chiếm | Đổi port hoặc kill process: `netstat -ano \| findstr :8000` rồi `taskkill /PID <PID> /F` |
| `venv: command not found` | Chưa kích hoạt venv | Chạy `.\venv\Scripts\activate` |
| `npm: command not found` | Chưa cài Node.js | Tải và cài Node.js từ nodejs.org |
