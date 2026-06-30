# 🚀 HOW TO RUN THE TRAINING PLATFORM

## Prerequisites
- Python 3.11 with PyTorch, scikit-learn, FastAPI
- API server running on http://localhost:8000

## Quick Start (3 Steps)

### Step 1: Start the API Server
Open PowerShell and run:
```powershell
cd d:\Nam4\ThucTap\EBM_SVM\api_base
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Step 2: Open Web Interface
Open your browser and navigate to:
```
file:///d:/Nam4/ThucTap/EBM_SVM/training_demo.html
```

You should see:
- **API Status**: ONLINE (green)
- **Upload Area**: Drag-drop zone for CSV files
- **Configuration**: Adjust training parameters
- **Latest Results**: Shows Iris dataset (100% accuracy)

### Step 3: Train Your Dataset
1. Click upload area or drag CSV file
2. (Optional) Adjust parameters:
   - Test Size: 5-50%
   - EBM Epochs: 10-100 (lower = faster training)
3. Click "Train Model"
4. Watch progress bar
5. View results when complete

## 📁 File Locations

| File | Purpose |
|------|---------|
| `training_demo.html` | Main web interface |
| `api_base/app/routers/train_endpoint.py` | Training endpoint |
| `data/Iris.csv` | Example dataset |

## 🧪 Test the API Directly

### Test 1: Check API is Running
```bash
python -c "import requests; print(requests.get('http://localhost:8000/api/ensemble/info').status_code)"
```
Expected: `200`

## 🎯 Using Your Own Datasets

### CSV Format Requirements
Your CSV file should have:
- One row per sample
- Features in columns (numerical or categorical)
- **Last column = target/label**

Example:
```
feature1, feature2, feature3, target
5.1,      3.5,      1.4,      setosa
7.0,      3.2,      4.7,      versicolor
6.3,      3.3,      6.0,      virginica
```

### Supported Data Types
- ✅ Numerical features (float, int)
- ✅ Categorical features (string, auto-encoded)
- ✅ Binary classification (2 classes)
- ✅ Multi-class classification (3+ classes)
- ✅ Any dataset size (but training time increases)

### Data Preprocessing (Automatic)
- Missing values: None (assume clean data)
- Categorical encoding: OneHotEncoder
- Numerical scaling: StandardScaler (per-feature normalization, inside ensemble)
- Feature reduction: PCA (95% variance default)
- Data split: (1-test_size)*75% train, (1-test_size)*25% validation, test_size% test

## ⚙️ Configuration Options

### Training Parameters (Web Interface)

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| Test Size | 20% | Fraction of data for testing |
| EBM Epochs | 30 | How many times model trains on data (higher = better but slower) |
| PCA Variance | 95% | How much information to keep after compression (95% = remove 5% least important features) |
| EBM Weight | 0.5 | Deprecated — kept for backward compatibility |

### Tips for Configuration
- **Small datasets** (< 1000 samples): Use 10-50 epochs
- **Large datasets** (> 10000 samples): Use 50-100 epochs
- **For faster training**: Lower epochs to 10-20
- **For best accuracy**: Increase epochs to 100+

## 📊 Results Interpretation

### Accuracy Metrics
- **SVM Accuracy**: Pure Support Vector Machine performance
- **EBM Accuracy**: Pure Energy-Based Model performance  
- **Ensemble Accuracy**: Enhanced SVM (PCA + EBM embeddings)
- **Final Accuracy**: Best of the three (SVM, EBM, Ensemble)
- **Best Model**: Which model achieved final accuracy
- **Improvement**: Gain vs baseline SVM

### Example Results
```
Dataset: Iris
Samples: 150
Features: 4
Classes: 3
Training Time: 4 seconds

SVM Accuracy:      100.00%
EBM Accuracy:      100.00%
Ensemble Accuracy: 100.00%
Final Accuracy:    100.00%
Improvement:       +0.00% (already perfect!)
PCA Components:    3 (kept 3 out of 4 features)
```

### Accuracy Ranges
- **90-100%**: Excellent - use with confidence
- **80-90%**: Good - acceptable for most applications
- **70-80%**: Fair - consider more data or feature engineering
- **< 70%**: Poor - review data quality or try different parameters

## 🔧 Troubleshooting

### Issue: "API not responding" or "Connection refused"
**Solution**: Make sure API server is running
```powershell
cd d:\Nam4\ThucTap\EBM_SVM\api_base
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Issue: "Port 8000 already in use"
**Solution**: Kill existing process and restart
```powershell
netstat -ano | findstr :8000
taskkill /PID {PID_NUMBER} /F
```

### Issue: "Training takes too long"
**Solution**: Reduce epochs or dataset size
- Reduce EBM Epochs to 10-20
- Use a subset of your data
- Check if dataset has many features (> 1000)

### Issue: "CSV file not found" or "File format error"
**Solution**: Ensure CSV is properly formatted
- File must be named `*.csv`
- All rows must have same number of columns
- No empty cells in target column
- Target column must be last column

### Issue: "Only 1 member in class X"
**Solution**: Your dataset has imbalanced or very small classes
- Ensure each class has at least 2 samples
- Remove very rare classes if needed
- Increase dataset size

## 📈 Performance Benchmarks

| Dataset | Samples | Features | Classes | Time | Accuracy |
|---------|---------|----------|---------|------|----------|
| Iris | 150 | 4 | 3 | ~4s | 100% |
| Wine | 178 | 13 | 3 | ~3s | 100% |
| Breast Cancer | 569 | 30 | 2 | ~15s | 97.4% |
| Adult (truncated) | 1000 | 14 | 2 | ~30s | 81-85% |

Training time depends on:
- Dataset size (more samples = longer)
- Number of features (more features = longer)
- EBM epochs (more epochs = longer)
- System specs (CPU speed, RAM)

## 💾 Saving Your Results

### Automatic Save
Results are displayed in the web interface immediately after training.

### Export as JSON
Click "Download Results (JSON)" to save detailed metrics:
```json
{
  "status": "success",
  "dataset_name": "your_dataset",
  "samples": 150,
  "features": 4,
  "classes": 3,
  "svm_accuracy": 0.95,
  "ebm_accuracy": 0.92,
  "ensemble_accuracy": 0.96,
  "final_accuracy": 0.96,
  "improvement": 0.01,
  "pca_components": 3
}
```

### Store in Database (Future)
Results can be stored in PostgreSQL for historical tracking.

## 🔐 Security Notes

- API runs locally (localhost:8000) - safe from external access
- No data uploaded to cloud - all processing local
- CSV files processed in memory - no storage
- Results stored only in browser/JSON exports

## 📞 Support & Debugging

### Check API Status
```bash
curl http://localhost:8000/api/ensemble/
```

### View API Logs
Logs appear in terminal where API was started.

### View Raw API Response
Open browser console (F12) and run:
```javascript
fetch('http://localhost:8000/api/ensemble/info')
  .then(r => r.json())
  .then(data => console.log(JSON.stringify(data, null, 2)))
```

## 🎓 Learn More

### About the Models
- **EBM**: Neural network that learns an energy function E(x, y) — low energy = confident
- **SVM**: Classical ML classifier with RBF kernel, trained on PCA features
- **Ensemble**: Feature Augmentation — EBM embeddings (penultimate layer) are concatenated with PCA features, then SVM is trained on this augmented set

### Configuration Deep-Dive
- PCA reduces features while keeping important information
- Stratified splits ensure balanced class distribution
- StandardScaler ensures features are comparable scale

### Advanced Usage
For developers, see:
- `api_base/ml_models/energy_based_model.py` - EBM implementation
- `api_base/ml_models/ebm_svm_ensemble_v3.py` - Ensemble logic
- `api_base/app/routers/train_endpoint.py` - API handler

## ✅ Checklist Before Starting

- [ ] Python 3.11 installed
- [ ] API requirements installed (`pip install fastapi uvicorn torch scikit-learn pandas`)
- [ ] CSV file ready in `/data/` folder
- [ ] Port 8000 is available (not in use)
- [ ] Browser updated (modern JavaScript support)

## 🎉 Ready to Go!

You're all set! Start the API server, open the web interface, and begin training models on your datasets.

**Questions? Check the troubleshooting section above or review the source code comments.**
