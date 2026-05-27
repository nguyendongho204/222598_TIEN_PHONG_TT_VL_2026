import pandas as pd

# Đọc file lớn
input_path = 'EBM_SVM/data/creditcard.csv'
output_path = 'EBM_SVM/data/creditcard_sample.csv'

# Lấy 1000 dòng đầu
chunk_size = 1000

df = pd.read_csv(input_path)
df_sample = df.head(chunk_size)
df_sample.to_csv(output_path, index=False)
print(f"Đã tạo file {output_path} với {chunk_size} dòng đầu.")
