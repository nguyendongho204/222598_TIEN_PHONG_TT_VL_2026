import pandas as pd
from sklearn.datasets import make_classification, make_moons

# Tạo dữ liệu phi tuyến, nhiều chiều, nhiều nhiễu
X1, y1 = make_moons(n_samples=2000, noise=0.4, random_state=42)
X2, y2 = make_classification(n_samples=2000, n_features=20, n_informative=10, n_redundant=5, n_clusters_per_class=2, flip_y=0.2, class_sep=0.5, random_state=42)

# Lưu dữ liệu moons
df_moons = pd.DataFrame(X1, columns=[f"f{i+1}" for i in range(X1.shape[1])])
df_moons['Class'] = y1
df_moons.to_csv('EBM_SVM/data/moons_complex.csv', index=False)

# Lưu dữ liệu classification
df_class = pd.DataFrame(X2, columns=[f"f{i+1}" for i in range(X2.shape[1])])
df_class['Class'] = y2
df_class.to_csv('EBM_SVM/data/classification_complex.csv', index=False)

print("Đã tạo moons_complex.csv và classification_complex.csv với dữ liệu phi tuyến, nhiều chiều, nhiều nhiễu.")
