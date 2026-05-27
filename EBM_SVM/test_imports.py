"""
Test imports to ensure all modules load correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "api_base"))

print("\n" + "="*60)
print("TESTING MODULE IMPORTS")
print("="*60)

try:
    print("\n[1/4] Importing ProvenPCASVMEnsemble...")
    from ml_models.proven_ensemble import ProvenPCASVMEnsemble
    print("  ✓ Success")
    
    print("\n[2/4] Importing UniversalEBMSVMPipeline...")
    from ml_models.universal_pipeline import UniversalEBMSVMPipeline
    print("  ✓ Success")
    
    print("\n[3/4] Importing universal_pipeline router...")
    from app.routers import universal_pipeline
    print("  ✓ Success")
    print(f"  Router: {universal_pipeline.router}")
    
    print("\n[4/4] Importing FastAPI app...")
    from app.main import app
    print("  ✓ Success")
    print(f"  App: {app}")
    print(f"  Routes: {[route.path for route in app.routes if 'universal' in route.path]}")
    
    print("\n" + "="*60)
    print("ALL IMPORTS SUCCESSFUL ✓")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\n❌ Import Error: {str(e)}")
    import traceback
    traceback.print_exc()
