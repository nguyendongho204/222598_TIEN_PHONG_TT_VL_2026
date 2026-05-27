
# Thư viện xử lý bảng
import pandas as pd
# Các hàm sinh dữ liệu mẫu từ sklearn
from sklearn.datasets import make_moons, make_circles, load_wine, load_breast_cancer, make_classification

# Tạo dữ liệu make_classification (nhiều chiều, phi tuyến, nhiều nhiễu)
X_mc, y_mc = make_classification(
	n_samples=1000,         # Số mẫu
	n_features=8,           # Số đặc trưng
	n_informative=6,        # Số đặc trưng thực sự hữu ích
	n_redundant=2,          # Số đặc trưng dư thừa
	n_clusters_per_class=2, # Số cụm mỗi lớp
	n_classes=2,            # Số lớp
	flip_y=0.2,             # Tỉ lệ nhiễu nhãn
	class_sep=0.5,          # Độ tách biệt giữa các lớp
	random_state=42         # Seed cố định
)
df_mc = pd.DataFrame(X_mc, columns=[f'feature{i+1}' for i in range(X_mc.shape[1])])
df_mc['label'] = y_mc
df_mc.to_csv('data/make_classification.csv', index=False)

# Tạo dữ liệu make_moons (2 lớp hình mặt trăng)
X_moons, y_moons = make_moons(n_samples=500, noise=0.25, random_state=42)
df_moons = pd.DataFrame(X_moons, columns=['feature1', 'feature2'])
df_moons['label'] = y_moons
df_moons.to_csv('data/moons.csv', index=False)

# Tạo dữ liệu make_moons nhiều noise
X_moons_noise, y_moons_noise = make_moons(n_samples=500, noise=0.45, random_state=42)
df_moons_noise = pd.DataFrame(X_moons_noise, columns=['feature1', 'feature2'])
df_moons_noise['label'] = y_moons_noise
df_moons_noise.to_csv('data/moons_noise.csv', index=False)

# Tạo dữ liệu make_circles (2 lớp hình vòng tròn)
X_circles, y_circles = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
df_circles = pd.DataFrame(X_circles, columns=['feature1', 'feature2'])
df_circles['label'] = y_circles
df_circles.to_csv('data/circles.csv', index=False)

# Tạo dữ liệu Wine (phân loại rượu vang)
wine = load_wine(as_frame=True)
df_wine = wine.data.copy()
df_wine['label'] = wine.target
df_wine.to_csv('data/wine.csv', index=False)

# Tạo dữ liệu Breast Cancer (phân loại ung thư vú)
cancer = load_breast_cancer(as_frame=True)
df_cancer = cancer.data.copy()
df_cancer['label'] = cancer.target
df_cancer.to_csv('data/breast_cancer.csv', index=False)
