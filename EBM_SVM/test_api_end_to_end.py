#!/usr/bin/env python
"""
End-to-end API test: Upload file + Compare EBM-SVM vs Baseline
"""
import requests
import json
import time
from pathlib import Path

API_URL = "http://localhost:8000"
FILE_PATH = "data/adult.csv"

print("=" * 80)
print("END-TO-END API TEST: Upload + Compare")
print("=" * 80)

# Step 1: Upload file
print("\n[1] Uploading file...")
with open(FILE_PATH, 'rb') as f:
    files = {'file': (Path(FILE_PATH).name, f)}
    response = requests.post(f"{API_URL}/api/files/upload", files=files)

print(f"Status: {response.status_code}")
data = response.json()
print(json.dumps(data, indent=2))

if response.status_code != 200:
    print("❌ Upload failed!")
    exit(1)

file_id = data.get('file_id')
print(f"✓ File ID: {file_id}")

# Step 2: Call Universal Pipeline Compare endpoint
print("\n[2] Calling /api/universal/compare...")
payload = {
    "file_id": file_id,
    "test_size": 0.2,
    "device": "cpu"
}

start_time = time.time()
response = requests.post(f"{API_URL}/api/universal/compare", json=payload)
elapsed = time.time() - start_time

print(f"Status: {response.status_code}")
print(f"Time: {elapsed:.2f}s")

result = response.json()
print(json.dumps(result, indent=2))

# Step 3: Parse results
print("\n[3] RESULTS:")
print("=" * 80)
if response.status_code == 200:
    print(f"Dataset: {result.get('dataset_name')}")
    print(f"Samples: {result.get('n_samples')}, Features: {result.get('n_features')}")
    print()
    print(f"Baseline Accuracy:  {result.get('baseline_accuracy')*100:.2f}%")
    print(f"Optimized Accuracy: {result.get('optimized_accuracy')*100:.2f}%")
    print(f"Improvement:        +{result.get('improvement_pct'):.2f}%")
    print()
    print(f"Analysis time:      {result.get('analysis_time'):.2f}s")
    print(f"Baseline train:     {result.get('baseline_train_time'):.2f}s")
    print(f"EBM train:          {result.get('ebm_train_time'):.2f}s")
    print(f"SVM tuning:         {result.get('tuning_time'):.2f}s")
    print(f"Total time:         {result.get('total_time'):.2f}s")
    print("=" * 80)
    print("✓ SUCCESS!")
else:
    print(f"❌ FAILED!")
    print(result)
