#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test pipeline trực tiếp với adult.csv"""
import requests
import json
import time
from pathlib import Path
import sys
import io

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_URL = "http://localhost:8000"

print("\n" + "="*80)
print("TEST TRỰC TIẾP: adult.csv")
print("="*80)

# Kiểm tra API
print("\nKiểm tra API...")
try:
    r = requests.get(f"{API_URL}/api/health", timeout=5)
    if r.status_code == 200:
        print("✓ API ready!")
    else:
        print("❌ API không response")
        sys.exit(1)
except Exception as e:
    print(f"❌ API không accessible: {e}")
    sys.exit(1)

# Upload file
print("\n[1] Uploading adult.csv...")
try:
    with open("data/adult.csv", 'rb') as f:
        files = {'file': ('adult.csv', f)}
        r = requests.post(
            f"{API_URL}/api/files/upload",
            files=files,
            timeout=60
        )
    print(f"Upload Status: {r.status_code}")
    if r.status_code != 200:
        print(f"Error: {r.text}")
        sys.exit(1)
    
    data = r.json()
    file_id = data.get('file_id')
    print(f"✓ File ID: {file_id}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    sys.exit(1)

# Call compare endpoint
print("\n[2] Calling /api/universal/compare (Timeout: 600s)...")
payload = {
    "file_id": file_id,
    "test_size": 0.2,
    "device": "cpu"
}

start_time = time.time()
try:
    r = requests.post(
        f"{API_URL}/api/universal/compare",
        json=payload,
        timeout=600
    )
    elapsed = time.time() - start_time
    
    print(f"Status: {r.status_code}")
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} phút)")
    
    if r.status_code == 200:
        result = r.json()
        print("\n✓ SUCCESS!")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Error: {r.text}")
        
except requests.exceptions.Timeout:
    elapsed = time.time() - start_time
    print(f"❌ Timeout sau {elapsed:.2f}s ({elapsed/60:.1f} phút)")
except Exception as e:
    print(f"❌ Error: {e}")
