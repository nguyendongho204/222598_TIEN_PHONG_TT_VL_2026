

run backend:
cd d:\Nam4\ThucTap\EBM_SVM
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
venv\Scripts\python.exe ebm_svm_integration.py

cd d:\Nam4\ThucTap\EBM_SVM
venv\Scripts\python.exe backend.py

run frontend:
cd d:\Nam4\ThucTap\EBM_SVM\frontend
npm start


kaggle datasets download -d uciml/iris



venv\Scripts\python.exe ebm_svm_integration.py