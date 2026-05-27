
# Thư viện vẽ biểu đồ
import matplotlib.pyplot as plt
# Thư viện xử lý số liệu
import numpy as np
# Thư viện xử lý dữ liệu bảng (không dùng trong hàm chính)
import pandas as pd
# SVM từ sklearn
from sklearn import svm
# Hàm chia train/test
from sklearn.model_selection import train_test_split
# Chuẩn hóa dữ liệu
from sklearn.preprocessing import StandardScaler
# Sinh dữ liệu vòng tròn (circles)
from sklearn.datasets import make_circles



# Hàm vẽ ranh giới quyết định của mô hình SVM hoặc EBM-SVM
def plot_decision_boundary(clf, X, y, ax, title, feature_transform=None):
    h = .02  # Bước lưới nhỏ để vẽ mịn
    # Xác định phạm vi trục x, y
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # Tạo lưới điểm để dự đoán
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Nếu có hàm biến đổi đặc trưng (EBM), áp dụng vào lưới
    if feature_transform is not None:
        grid = feature_transform(grid)
    # Dự đoán nhãn cho từng điểm trên lưới
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    # Vẽ vùng phân loại
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # Vẽ các điểm dữ liệu thật
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(title)
    return scatter


# Hàm chính: So sánh SVM và EBM-SVM trên dữ liệu vòng tròn
def plot_circles_svm_ebm():
    # Sinh dữ liệu vòng tròn (circles) gồm 2 lớp
    X, y = make_circles(n_samples=400, noise=0.05, factor=0.5, random_state=42)
    # Chia train/test (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Chuẩn hóa dữ liệu về mean=0, std=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện SVM tuyến tính trên dữ liệu gốc
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(X_train_scaled, y_train)

    # Hàm giả lập embedding EBM: nối thêm sin/cos vào đặc trưng
    def ebm_transform(X):
        return np.hstack([X, np.sin(X), np.cos(X)])
    # Tạo đặc trưng mới cho train/test
    X_train_ebm = ebm_transform(X_train_scaled)
    X_test_ebm = ebm_transform(X_test_scaled)
    # Huấn luyện SVM trên đặc trưng EBM
    clf_ebm_svm = svm.SVC(kernel='linear')
    clf_ebm_svm.fit(X_train_ebm, y_train)

    # Vẽ hình so sánh ranh giới quyết định
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(clf_svm, X_train_scaled, y_train, axes[0], 'SVM tuyến tính')
    plot_decision_boundary(clf_ebm_svm, X_train_scaled, y_train, axes[1], 'SVM-EBM (biến đổi đặc trưng)', feature_transform=ebm_transform)
    plt.suptitle('So sánh ranh giới quyết định trên bộ dữ liệu circles')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('circles_svm_vs_ebm.png', dpi=150)
    plt.show()


# Để chạy vẽ hình minh họa, bỏ comment dòng sau:
plot_circles_svm_ebm()  # Chạy hàm chính để vẽ so sánh SVM và EBM-SVM

# Dữ liệu tổng hợp từ báo cáo
results = [
    {"Dataset": "moons.csv", "SVM Accuracy": 95.0, "EBM-SVM Accuracy": 71.0, "SVM F1": 95.0, "EBM-SVM F1": 69.7},
    {"Dataset": "circles.csv", "SVM Accuracy": 95.0, "EBM-SVM Accuracy": 99.2},
    {"Dataset": "moons_noise.csv", "SVM Accuracy": 83.6, "EBM-SVM Accuracy": 65.8, "SVM F1": 83.6, "EBM-SVM F1": 65.5},
    {"Dataset": "make_classification.csv", "SVM Accuracy": 79.9, "EBM-SVM Accuracy": 54.9},
    {"Dataset": "wine.csv", "SVM Accuracy": 70.8, "EBM-SVM Accuracy": 71.9, "SVM F1": 69.1, "EBM-SVM F1": 69.9},
    {"Dataset": "breast_cancer.csv", "SVM Accuracy": 92.3, "EBM-SVM Accuracy": 90.9}
]
df = pd.DataFrame(results)

# Biểu đồ so sánh Accuracy
fig, ax = plt.subplots(figsize=(10,6))
bar_width = 0.35
index = np.arange(len(df))

ax.bar(index, df["SVM Accuracy"], bar_width, label="SVM Accuracy", color="#1976d2")
ax.bar(index + bar_width, df["EBM-SVM Accuracy"], bar_width, label="EBM-SVM Accuracy", color="#ff9800")

ax.set_xlabel('Dataset')
ax.set_ylabel('Accuracy (%)')
ax.set_title('So sánh Accuracy SVM vs EBM-SVM trên các bộ dữ liệu')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(df["Dataset"], rotation=20)
ax.legend()
plt.tight_layout()
plt.savefig("compare_accuracy.png", dpi=150)

# Biểu đồ so sánh F1-score (nếu có)
df_f1 = df.dropna(subset=["SVM F1", "EBM-SVM F1"], how="all")
if not df_f1.empty:
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.bar(index[:len(df_f1)], df_f1["SVM F1"], bar_width, label="SVM F1-score", color="#1976d2")
    ax2.bar(index[:len(df_f1)] + bar_width, df_f1["EBM-SVM F1"], bar_width, label="EBM-SVM F1-score", color="#ff9800")
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('F1-score (%)')
    ax2.set_title('So sánh F1-score SVM vs EBM-SVM trên các bộ dữ liệu')
    ax2.set_xticks(index[:len(df_f1)] + bar_width / 2)
    ax2.set_xticklabels(df_f1["Dataset"], rotation=20)
    ax2.legend()
    plt.tight_layout()
    plt.savefig("compare_f1score.png", dpi=150)

print("Đã tạo compare_accuracy.png và compare_f1score.png (nếu có dữ liệu F1-score)")
