import time
import requests

API_BASE = 'http://localhost:8000'
file_path = 'data/adult.csv'

print('🔄 Tệp: adult.csv')
print('📊 Kích thước: 32,561 dòng → giới hạn API: 5,000 samples')
print()

# Upload file
print('[1] Upload file...')
start = time.time()
with open(file_path, 'rb') as f:
    files = {'file': f}
    r = requests.post(f'{API_BASE}/api/files/upload', files=files)
    file_id = r.json()['file_id']
upload_time = time.time() - start
print(f'    ✓ Upload: {upload_time:.2f}s')
print()

# Run universal compare
print('[2] Gọi /api/universal/compare...')
start = time.time()
r = requests.post(f'{API_BASE}/api/universal/compare', json={'file_id': file_id})
total_time = time.time() - start

if r.status_code == 200:
    data = r.json()
    print(f'    ✓ Hoàn thành: {total_time:.2f}s')
    train_time = data.get('training_time_seconds', 'N/A')
    svm_acc = data.get('svm_accuracy', 'N/A')
    ebm_acc = data.get('ebm_svm_accuracy', 'N/A')
    print(f'    - Training time: {train_time}s')
    print(f'    - SVM Accuracy: {svm_acc}%')
    print(f'    - EBM-SVM Accuracy: {ebm_acc}%')
else:
    print(f'    ✗ Lỗi: {r.status_code}')
    print(f'    {r.text[:200]}')
