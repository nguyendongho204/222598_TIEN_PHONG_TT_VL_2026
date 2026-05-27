
# Thư viện xử lý bảng
import pandas as pd
# SVM từ sklearn
from sklearn.svm import SVC
# Đánh giá mô hình
from sklearn.metrics import classification_report, confusion_matrix

# Đọc dữ liệu train/test từ file csv
train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')

# Tách features và nhãn cho train/test
X_train = train.drop('label', axis=1)
y_train = train['label']
X_test = test.drop('label', axis=1)
y_test = test['label']

# Khởi tạo và huấn luyện mô hình SVM kernel RBF
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)

# Dự đoán nhãn cho tập test
preds = model.predict(X_test)

# Hiển thị 5 dòng đầu tiên của tập test với nhãn thật và dự đoán
first5 = X_test.head().copy()
first5['true_label'] = y_test.reset_index(drop=True).head()
first5['predicted'] = preds[:5]
print('--- First 5 test rows (features..., true_label, predicted) ---')
print(first5.to_string(index=False))

# In ma trận nhầm lẫn
cm = confusion_matrix(y_test, preds)
print('\n--- Confusion matrix (rows=true, cols=predicted) ---')
print(cm)

# In báo cáo phân loại
print('\n--- Classification report (test) ---')
print(classification_report(y_test, preds))
