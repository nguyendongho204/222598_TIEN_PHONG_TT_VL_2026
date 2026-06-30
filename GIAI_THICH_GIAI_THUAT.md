# Giải Thích Giải Thuật EBM + SVM Ensemble (v3 → v8)

## 1. Bài Toán

**Phân loại dữ liệu (Classification)**: Đầu vào là các đặc trưng (features), đầu ra là nhãn (class).

Ví dụ: Với 4 đặc trưng của hoa Iris (chiều dài đài, chiều rộng đài, chiều dài cánh, chiều rộng cánh), dự đoán loài hoa (Setosa, Versicolor, Virginica).

---

## 2. Ý Tưởng Tổng Quan

Hệ thống kết hợp **2 mô hình khác nhau** để tận dụng ưu điểm của cả hai:

| Mô hình | Bản chất | Điểm mạnh |
|---------|----------|-----------|
| **EBM** (Energy-Based Model) | Mạng neural nhân tạo (PyTorch) | Học được các quan hệ phức tạp, phi tuyến tính |
| **SVM** (Support Vector Machine) | Máy vector hỗ trợ (scikit-learn) | Ổn định, ít overfitting, ranh giới quyết định rõ ràng |

**Nguyên lý**: Hai mô hình có "cách nhìn" khác nhau về dữ liệu. Kết hợp chúng lại cho kết quả chính xác hơn từng mô hình riêng lẻ.

---

## 3. Pipeline Tổng Thể (Cách Huấn Luyện)

```
Dữ liệu CSV
    │
    ▼
┌─────────────────┐
│ 1. Scale dữ liệu │ ← StandardScaler: đưa về cùng thang đo
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ PCA    │ │ EBM    │
│ (giảm  │ │ (học   │
│  chiều)│ │  trên  │
│        │ │  gốc)  │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌────────┐ ┌────────┐
│ SVM    │ │ EBM    │
│ (học   │ │ (xuất  │
│  trên  │ │ năng   │
│  PCA)  │ │ lượng) │
└───┬────┘ └───┬────┘
    │          │
    └────┬─────┘
         ▼
  ┌──────────┐
  │ Ensemble │ ← Kết hợp theo chiến lược
  └────┬─────┘
       ▼
  ┌──────────┐
  │ Kết quả  │ ← Accuracy, Precision, Recall, F1
  └──────────┘
```

### Chi tiết từng bước

| Bước | Xử lý | Mục đích |
|------|-------|----------|
| 1 | **StandardScaler** | Đưa features về cùng khoảng giá trị (mean=0, std=1) — tránh feature có giá trị lớn áp đảo |
| 2 | **PCA** (Principal Component Analysis) | Tổng hợp các features tương đồng thành số lượng nhỏ hơn (vd: 100 → ~20), giữ 95% thông tin. Giúp SVM chạy nhanh, giảm nhiễu |
| 3 | **Train EBM** (trên dữ liệu gốc) | EBM là neural network → cần dữ liệu gốc đầy đủ để học các pattern phi tuyến. PCA có thể làm mất thông tin phi tuyến |
| 4 | **Train SVM** (trên dữ liệu PCA) | SVM RBF hoạt động tốt trên không gian đã giảm chiều, tìm ranh giới quyết định tối ưu |
| 5 | **Ensemble strategy** | Kết hợp EBM + SVM theo 1 trong 6 cách (v3-v8) |
| 6 | **Đánh giá** | So sánh accuracy của SVM đơn thuần vs EBM đơn thuần vs Ensemble |

---

## 4. EBM Hoạt Động Thế Nào?

EBM là mạng neural truyền thẳng (feedforward) với kiến trúc:

```
Input (features + one-hot label)
    │
Linear(256) + ReLU
    │
Linear(128) + ReLU
    │
Linear(64) + ReLU
    │
Linear(32)          ← Lớp này trích xuất embedding (đặc trưng trung gian)
    │
Linear(1)           ← Đầu ra: 1 giá trị năng lượng E(x,y)
```

**Cách hoạt động**:
- EBM học một hàm **năng lượng** E(x,y): với mẫu x và nhãn dự đoán y
- Nếu (x, y) là cặp **đúng**: năng lượng **thấp**
- Nếu (x, y) là cặp **sai**: năng lượng **cao**
- Khi dự đoán: EBM tính năng lượng cho từng class, chọn class có năng lượng thấp nhất

---

## 5. SVM Hoạt Động Thế Nào?

- Dùng **kernel RBF** (Radial Basis Function) để xử lý dữ liệu phi tuyến
- Tìm một **siêu phẳng** (hyperplane) phân tách các class với khoảng cách biên (margin) lớn nhất
- Tham số **C** được chọn qua cross-validation (thử [1, 10, 50, 100], chọn C tốt nhất)

---

## 6. So Sánh 6 Chiến Lược Ensemble (v3 → v8)

### V3 — Feature Augmentation (Global + Local Fusion)

**Cách hoạt động**:
1. Train EBM trên dữ liệu gốc → xuất ra energy scores E0, E1 và confidence C0, C1
2. Ghép [PCA features (20D) + EBM features (4D)] = 24D
3. Train SVM trên vector 24D này

**Giải thích**:
- EBM nhìn tổng thể (global) → nắm cấu trúc năng lượng toàn cục
- SVM RBF nhìn cục bộ (local) → tìm ranh giới chi tiết
- Kết hợp cả hai → bổ trợ lẫn nhau

---

### V4 — Stacking Ensemble

**Cách hoạt động**:
1. Train SVM và EBM độc lập
2. Lấy SVM probabilities (2 số) + EBM confidence (2 số) = 4 meta-features
3. Train LogisticRegression trên 4 meta-features này

**Giải thích**: LogisticRegression học cách phối hợp tối ưu giữa quyết định của SVM và EBM.

---

### V5 — Improved Stacking (Cải tiến từ V4)

**3 cải tiến chính so với V4**:

| Khía cạnh | V4 | V5 |
|-----------|----|----|
| **Đầu vào EBM** | Tất cả features | **SelectKBest** — chọn top 30 features quan trọng nhất (giảm nhiễu) |
| **Kích thước EBM** | [256, 128, 64] | **[512, 256, 128]** — mạng to hơn, học phức tạp hơn |
| **Meta features** | 4 (SVM probs + EBM conf) | **8** (thêm decision function + energies + energy gap) |
| **Meta classifier** | LogisticRegression (tuyến tính) | **GradientBoosting** (150 cây, phi tuyến) |

**Giải thích**: V5 mạnh nhất về độ chính xác vì dùng meta-classifier mạnh (GradientBoosting) và nhiều thông tin hơn.

---

### V6 — Feature Augmentation (Đơn giản hóa)

**Cách hoạt động**:
1. Train EBM nhỏ [64, 32] trên dữ liệu scale
2. Trích xuất 5 features EBM: energies (2) + softmax confidence (2) + predicted class (1)
3. Ghép [PCA + 5 EBM features] → train SVM duy nhất

**Giải thích**: Giống V3 nhưng đơn giản hơn, EBM nhỏ → nhanh hơn, phù hợp dataset lớn.

---

### V7 — Prediction Stacking

**Cách hoạt động**:
1. Lấy xác suất đầu ra từ SVM (2 số) và EBM (2 số)
2. Ghép = 4 con số
3. LogisticRegression học trên 4 con số này để quyết định cuối

**Giải thích**: Chỉ dùng xác suất dự đoán (không dùng raw features), rất nhẹ, phù hợp hệ thống real-time.

---

### V8 — Confidence Cascade (Mới nhất)

**Cách hoạt động**:
1. SVM dự đoán trước, kèm xác suất tin tưởng (confidence)
2. Nếu confidence >= ngưỡng (tìm tối ưu bằng grid search 0.5→0.95) → **lấy kết quả SVM**
3. Nếu confidence < ngưỡng → **EBM quyết định thay**

**Giải thích toán học**:

```
P_SVM(x) = max(proba_SVM(x))   // độ tự tin của SVM

Nếu P_SVM(x) >= threshold → y = y_SVM
Nếu P_SVM(x) <  threshold → y = y_EBM

Với threshold tối ưu = argmax_{t ∈ [0.5,0.95]} Accuracy_cascade(t)
```

**Giải thích trực quan**:
- SVM xử lý các mẫu "dễ" (rõ ràng)
- EBM xử lý các mẫu "khó" (mơ hồ, biên giới)
- Ngưỡng được tự động tìm ra để tối ưu độ chính xác tổng thể

---

## 7. Bảng So Sánh Tổng Hợp

| Version | Chiến lược | Số meta-features | Meta-classifier | EBM size | Điểm mạnh |
|---------|-----------|:----------------:|:---------------:|:--------:|-----------|
| **V3** | Feature Augmentation | 4 | SVM (RBF) | [256,128,64] | Global+Local fusion |
| **V4** | Stacking | 4 | LogisticRegression | [256,128,64] | Đơn giản, dễ hiểu |
| **V5** | Stacking cải tiến | 8 | GradientBoosting | [512,256,128] | **Accuracy cao nhất** |
| **V6** | Feature Augmentation | 5 | SVM (RBF) | [64,32] | Nhanh, nhẹ |
| **V7** | Prediction Stacking | 4 | LogisticRegression | [64,32] | Real-time, scalable |
| **V8** | Confidence Cascade | 0 (threshold) | Cascade rule | [64,32] | **Thông minh nhất**, xử lý vùng mờ |

---

## 8. Chọn Phiên Bản Nào Cho Dataset Nào?

| Loại dataset | Khuyên dùng | Giải thích ngắn gọn |
|-------------|:-----------:|---------------------|
| **Nhỏ, ít features** (< 1000 mẫu, < 10 features) | **V3** | EBM sâu tận dụng non-linear patterns tốt |
| **Nhiều features, nhiễu** (> 50 features) | **V5** | SelectKBest lọc nhiễu, GradientBoosting mạnh |
| **Mất cân bằng lớp** | **V8** | Cascade linh hoạt theo từng mẫu |
| **Lớn** (> 10.000 mẫu) | **V6 hoặc V7** | EBM nhỏ, train nhanh |
| **Multi-class** (> 2 lớp) | **V3 hoặc V5** | EBM multi-class energy hiệu quả |
| **Cần giải thích kết quả** | **V4** | LogisticRegression dễ hiểu |
| **Yêu cầu accuracy cao nhất** | **V5** | Stacking với GradientBoosting mạnh nhất |

---

## 9. Kết Luận

1. **EBM** (neural network) và **SVM** (kernel) bổ trợ cho nhau
2. **V3→V8** là các chiến lược kết hợp khác nhau, mỗi phiên bản cải tiến dựa trên hạn chế của phiên bản trước
3. **Không có phiên bản tốt nhất tuyệt đối** — tùy thuộc vào đặc điểm dữ liệu (kích thước, số chiều, độ nhiễu, độ mất cân bằng)
4. Đóng góp chính: Chứng minh ensemble giữa energy-based model và kernel method có thể cải thiện độ chính xác so với từng mô hình riêng lẻ

---

*Tài liệu được tạo tự động từ mã nguồn dự án EBM+SVM Ensemble*
