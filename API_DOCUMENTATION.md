# 📚 EBM+SVM API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Root Endpoint
```
GET /api/ensemble/
```
Returns service information and available endpoints.

**Response:**
```json
{
  "service": "EBM + SVM Ensemble API",
  "version": "3.0",
  "endpoints": {
    "GET /api/ensemble/": "This message",
    "GET /api/ensemble/info": "Get model information",
    "POST /api/ensemble/predict": "Make predictions",
    "POST /api/ensemble/health": "Health check",
    "POST /api/ensemble/train": "Train model (stub)",
    "POST /api/train-dataset": "Train on uploaded CSV"
  }
}
```

---

### 2. Model Info
```
GET /api/ensemble/info
```
Get current model configuration and status.

**Response:**
```json
{
  "status": "success",
  "loaded": true,
  "model_type": "EBM + SVM Ensemble V3",
  "info": {
    "pca_variance": 0.95,
    "ebm_epochs": 100,
    "ebm_hidden_dims": [256, 128, 64, 32],
    "svm_C_values": [1.0, 10.0, 50.0, 100.0],
    "device": "cpu",
    "model_trained": false,
    "ensemble_strategy": "Feature Augmentation (EBM features + SVM)"
  }
}
```

---

### 3. Health Check
```
POST /api/ensemble/health
```
Check if API is running and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_type": "EBM + SVM Ensemble V3",
  "model_trained": false,
  "timestamp": "2026-05-31T09:01:43"
}
```

---

### 4. Make Predictions
```
POST /api/ensemble/predict
```
Make predictions using the trained model.

**Request Body:**
```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.9, 4.3, 1.3]
  ],
  "description": "Iris samples"
}
```

**Response (if model trained):**
```json
{
  "predictions": [0, 1],
  "confidences": [[0.95, 0.05], [0.13, 0.87]],
  "ensemble_info": {
    "model_type": "EBM + SVM Ensemble V3",
    "num_samples": 2,
    "num_features": 4,
    "ensemble_strategy": "Feature Augmentation",
    "description": "Iris samples"
  }
}
```

**Response (if model not trained):**
```json
{
  "detail": "Model not trained yet. Train the model first."
}
```

---

### 5. Train Dataset (NEW)
```
POST /api/train-dataset
```
**⭐ MAIN ENDPOINT** - Upload CSV file and train model

**Request:**
- **Method**: POST
- **Content-Type**: multipart/form-data

**Form Parameters:**

| Name | Type | Default | Required | Description |
|------|------|---------|----------|-------------|
| `file` | File (CSV) | - | ✅ Yes | CSV dataset file |
| `test_size` | float | 0.2 | ❌ No | Test set fraction (0.05-0.5) |
| `ebm_epochs` | int | 100 | ❌ No | EBM training iterations (10-500) |
| `pca_variance` | float | 0.95 | ❌ No | PCA variance threshold (0.50-0.99) |
| `ebm_weight` | float | 0.5 | ❌ No | Deprecated — kept for backward compatibility |

**Example Request (cURL):**
```bash
curl -X POST http://localhost:8000/api/train-dataset \
  -F "file=@data/Iris.csv" \
  -F "test_size=0.2" \
  -F "ebm_epochs=50" \
  -F "pca_variance=0.95"
```

**Example Request (Python):**
```python
import requests

with open('data/Iris.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'test_size': 0.2,
        'ebm_epochs': 50,
        'pca_variance': 0.95
    }
    response = requests.post(
        'http://localhost:8000/api/train-dataset',
        files=files,
        data=data,
        timeout=600
    )
    result = response.json()
    print(result)
```

**Example Request (JavaScript):**
```javascript
const file = document.getElementById('fileInput').files[0];
const formData = new FormData();
formData.append('file', file);
formData.append('test_size', 0.2);
formData.append('ebm_epochs', 50);
formData.append('pca_variance', 0.95);

fetch('http://localhost:8000/api/train-dataset', {
    method: 'POST',
    body: formData,
    timeout: 600000
})
.then(r => r.json())
.then(data => console.log(data));
```

**Success Response:**
```json
{
  "status": "success",
  "dataset_name": "Iris",
  "samples": 150,
  "features": 3,
  "classes": 3,
  "svm_accuracy": 1.0,
  "ebm_accuracy": 1.0,
  "ensemble_accuracy": 1.0,
  "final_accuracy": 1.0,
  "best_model": "EBM+SVM Ensemble",
  "improvement": 0.0,
  "optimal_ebm_weight": 0.0,
  "pca_components": 3,
  "per_class_accuracy": [1.0, 1.0, 1.0],
  "confusion_matrix": [[...]],
  "classification_report": {...}
}
```

**Error Response:**
```json
{
  "status": "error",
  "error": "The least populated classes in y have only 1 member..."
}
```

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | OK - Request successful | Training completed |
| 400 | Bad Request | Missing required parameter |
| 422 | Validation Error | Invalid data type |
| 500 | Server Error | Unexpected error |

### Error Response Format
```json
{
  "detail": "Error description or list of validation errors"
}
```

### Validation Errors
```json
{
  "detail": [
    {
      "loc": ["body", "features"],
      "msg": "value is not a valid list",
      "type": "type_error.list"
    }
  ]
}
```

---

## Authentication & Security

- ✅ **Local Access Only**: Running on 127.0.0.1 (not exposed to internet)
- ✅ **No Authentication Required**: Safe for development use
- ✅ **CORS Enabled**: Allows requests from browser
- ⚠️ **Production Note**: Add authentication before deploying to production

---

## Rate Limiting

- ⚠️ Currently **No Rate Limiting**
- Recommendation: Implement in production
- Suggested: Max 10 requests/minute per IP

---

## Timeouts

- **Request Timeout**: 600 seconds (10 minutes) for training requests
- **Connection Timeout**: 30 seconds

---

## CSV File Format Requirements

### Accepted Format
```csv
feature1, feature2, feature3, target
5.1,      3.5,      1.4,      setosa
7.0,      3.2,      4.7,      versicolor
6.3,      3.3,      6.0,      virginica
```

### Requirements
- ✅ Headers optional (first row treated as data if no headers)
- ✅ Numerical features (float, int)
- ✅ Categorical features (string)
- ✅ **Last column must be target/label**
- ⚠️ No missing values (NaN, NULL)
- ⚠️ UTF-8 encoding recommended

### Examples

**Iris Dataset:**
```
SepalLength,SepalWidth,PetalLength,PetalWidth,Species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
```

**Binary Classification:**
```
age,income,credit_score,approved
25,45000,720,yes
45,95000,800,yes
30,55000,650,no
```

**Multi-Feature:**
```
temp,humidity,pressure,raining
72,45,1013,no
65,78,1009,yes
68,52,1015,no
```

---

## Response Headers

All responses include CORS headers:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

---

## Webhook Support

- ❌ Currently **Not Supported**
- Future Feature: Callback URL for training completion notifications

---

## API Versioning

Current Version: **3.0**
- Previous: 2.0 (basic ensemble)
- Previous: 1.0 (basic API)

Version specified in all responses: `"version": "3.0"`

---

## Performance Metrics

### Typical Response Times

| Operation | Time | Notes |
|-----------|------|-------|
| GET /api/ensemble/ | < 1ms | Instant |
| GET /api/ensemble/info | < 1ms | Configuration only |
| POST /api/ensemble/health | < 5ms | Status check |
| POST /api/train-dataset | 4-120s | Depends on dataset size |

### Resource Usage

| Metric | Value | Notes |
|--------|-------|-------|
| Memory (idle) | ~200 MB | After startup |
| Memory (training) | ~500-1000 MB | Depends on dataset |
| CPU (idle) | < 5% | Minimal load |
| CPU (training) | 80-100% | Single core |

---

## Logging

Logs appear in terminal where API was started:
```
2026-05-31 16:26:12,062 - app.routers.train_endpoint - INFO - [Dataset Training] Loading file: adult.csv
2026-05-31 16:26:12,103 - app.routers.train_endpoint - INFO - Dataset shape: (32561, 15)
2026-05-31 16:26:12,254 - app.routers.train_endpoint - INFO - Training model...
```

### Log Levels
- `INFO`: Normal operation
- `WARNING`: Potential issues
- `ERROR`: Operation failed
- `DEBUG`: Detailed debugging (if enabled)

---

## Testing

### Test All Endpoints
```bash
python test_api_training.py
```

### Quick Health Check
```bash
python -c "import requests; print('API Status:', requests.get('http://localhost:8000/api/ensemble/').status_code)"
```

### Load Test (Coming Soon)
```bash
locust -f locustfile.py --host=http://localhost:8000
```

---

## Best Practices

### ✅ DO
- Use appropriate `test_size` for your dataset (20-30% typical)
- Start with lower epochs (30-50) then increase if needed
- Use `pca_variance=0.95` for high-dimensional data
- Save results after training

### ❌ DON'T
- Send very large files (> 100MB) - may timeout
- Set `ebm_epochs` to extremely high values (> 1000)
- Use datasets with only 1 sample per class
- Train concurrently on same API (sequential recommended)

---

## Migration Guide

### From V2 to V3
- New endpoint: `/api/train-dataset` (CSV upload)
- Improved error handling with better messages
- Better support for small datasets
- Automatic stratification fallback

---

## Support & Issues

- 🐛 **Bug Report**: Check terminal logs for error messages
- 💡 **Feature Request**: Documented in future features list
- 📞 **Technical Help**: Review code comments in source files

---

## License & Attribution

- PyTorch: BSD License
- scikit-learn: BSD License
- FastAPI: MIT License

**Made with ❤️ for Machine Learning**
