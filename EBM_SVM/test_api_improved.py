#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end API test: Upload file + Compare EBM-SVM vs Baseline
Với timeout tối ưu hóa để tránh reset kết nối
"""
import requests
import json
import time
from pathlib import Path
import sys
import io

# Fix encoding issue on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_URL = "http://localhost:8000"

def test_dataset(file_path, timeout=600):
    """
    Test đầu cuối cho một dataset
    timeout: 600 giây (10 phút) cho dataset lớn
    """
    print("\n" + "=" * 80)
    print(f"TEST: {Path(file_path).name}")
    print("=" * 80)
    
    if not Path(file_path).exists():
        print(f"❌ File không tồn tại: {file_path}")
        return False
    
    # Step 1: Upload file
    print("\n[1] Uploading file...")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            response = requests.post(
                f"{API_URL}/api/files/upload",
                files=files,
                timeout=60  # Upload timeout
            )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
    
    print(f"Upload Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return False
    
    data = response.json()
    file_id = data.get('file_id')
    print(f"✓ File ID: {file_id}")
    
    # Step 2: Call Universal Pipeline Compare endpoint
    print("\n[2] Calling /api/universal/compare...")
    print(f"   (Timeout: {timeout}s)")
    
    payload = {
        "file_id": file_id,
        "test_size": 0.2,
        "device": "cpu"
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_URL}/api/universal/compare",
            json=payload,
            timeout=timeout  # Dài timeout để đợi training
        )
        elapsed = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} phút)")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ SUCCESS!")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"❌ Request Timeout sau {elapsed:.2f}s ({elapsed/60:.1f} phút)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Main
if __name__ == "__main__":
    print("\n" + "="*80)
    print("EBM-SVM PIPELINE - API END-TO-END TEST")
    print("="*80)
    
    # Check if API is running
    print("\nKiểm tra API...")
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is running!")
        else:
            print("❌ API không response")
            sys.exit(1)
    except Exception as e:
        print(f"❌ API không accessible: {e}")
        sys.exit(1)
    
    # Test with small dataset first
    print("\n\n" + "="*80)
    print("BƯỚC 1: Test với dataset nhỏ (Iris)")
    print("="*80)
    iris_ok = test_dataset("data/Iris.csv", timeout=120)
    
    # Test with larger dataset
    if iris_ok:
        print("\n\n" + "="*80)
        print("BƯỚC 2: Test với dataset lớn (adult.csv)")
        print("="*80)
        adult_ok = test_dataset("data/adult.csv", timeout=600)
        
        print("\n\n" + "="*80)
        print("KẾT QUẢ CUỐI CÙNG")
        print("="*80)
        print(f"Iris.csv: {'✓ PASSED' if iris_ok else '❌ FAILED'}")
        print(f"adult.csv: {'✓ PASSED' if adult_ok else '❌ FAILED'}")
        print("="*80)
    else:
        print("\n❌ Iris test failed, skipping adult.csv test")
