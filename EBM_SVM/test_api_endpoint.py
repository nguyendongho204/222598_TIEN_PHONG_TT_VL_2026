"""
Test API endpoint for universal pipeline
"""

import sys
from pathlib import Path
import json
import subprocess
import time
import requests

def test_api():
    """Test /api/universal/compare endpoint"""
    
    # Start API in background
    print("\n" + "="*60)
    print("STARTING BACKEND API")
    print("="*60)
    
    api_path = Path(__file__).parent / "api_base"
    print(f"Working dir: {api_path}")
    
    # Start server
    print("Starting Uvicorn server...")
    proc = subprocess.Popen(
        [sys.executable, "run_api.py"],
        cwd=str(api_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Test endpoint
    url = "http://localhost:8000/api/universal/compare"
    payload = {
        "file_id": "wine.csv",
        "test_size": 0.2,
        "device": "cpu"
    }
    
    try:
        print(f"\nTesting endpoint: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse:")
            print(json.dumps(result, indent=2))
            
            # Validate response
            assert result['status'] == 'success'
            assert 'baseline_accuracy' in result
            assert 'final_accuracy' in result
            assert result['final_accuracy'] >= result['baseline_accuracy']
            
            print(f"\n✓ API Test PASSED")
            return True
        else:
            print(f"\n❌ API Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Request Error: {str(e)}")
        return False
    finally:
        # Kill server
        print("\nStopping API server...")
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
