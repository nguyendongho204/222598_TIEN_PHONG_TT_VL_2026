import React, { useState, useEffect } from 'react';
import './UniversalPipeline.css';

function UniversalPipeline({ fileId, onResults, displayResults }) {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(displayResults || null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');

  // If displayResults prop changes, update internal state
  useEffect(() => {
    if (displayResults) {
      setResults(displayResults);
    }
  }, [displayResults]);

  const API_BASE = 'http://localhost:8000';

  const runComparison = async () => {
    console.log('runComparison called, fileId:', fileId);
    if (!fileId) {
      setError('Vui lòng upload file trước');
      console.error('Missing fileId:', fileId);
      return;
    }

    setLoading(true);
    setError(null);
    setProgress(0);
    setResults(null);

    try {
      setStatus('Bắt đầu so sánh...');
      const url = `${API_BASE}/api/universal/compare`;
      const payload = {
        file_id: fileId,
        test_size: 0.2,
        device: 'cpu'
      };
      console.log('API_BASE:', API_BASE);
      console.log('URL:', url);
      console.log('Payload:', payload);
      console.log('Calling API with fileId:', fileId);

      let response;
      try {
        response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            file_id: fileId,
            test_size: 0.2,
            device: 'cpu'
          })
        });
        console.log('Fetch succeeded, status:', response.status);
      } catch (fetchErr) {
        console.error('Fetch error:', fetchErr);
        throw new Error(`Network error: ${fetchErr.message}`);
      }

      if (!response.ok) {
        console.error('Response not ok, status:', response.status);
        try {
          const errorData = await response.json();
          throw new Error(errorData.detail || `API Error: ${response.status} ${response.statusText}`);
        } catch (parseError) {
          throw new Error(`API Error ${response.status}: ${response.statusText}`);
        }
      }

      const data = await response.json();
      
      setResults(data);
      setStatus('Hoàn tát!');
      setProgress(100);

      if (onResults) {
        onResults(data);
      }

    } catch (err) {
      console.error('Full error:', err);
      console.error('Error message:', err.message);
      console.error('Error type:', typeof err);
      const errorMsg = err.message || 'Lỗi so sánh không xác định';
      setError(errorMsg);
      setStatus('Lỗi!');
      alert('❌ API Error: ' + errorMsg);  // DEBUG: Show alert
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="universal-pipeline-container">
      <h2>🧪 Universal EBM-SVM Comparison</h2>
      
      <div className="pipeline-section">
        <button 
          onClick={runComparison} 
          disabled={loading || !fileId}
          className="run-button"
        >
          {loading ? '⏳ Đang chạy...' : '▶ So Sánh Baseline vs Adaptive EBM-SVM'}
        </button>

        {loading && (
          <div className="progress-section">
            <p>{status}</p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }}>
                {progress}%
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            ❌ Lỗi: {error}
          </div>
        )}

        {results && (
          <div className="results-section">
            <h3>📊 Kết Quả So Sánh EBM-SVM</h3>

            {/* Dataset Information */}
            <div className="dataset-info">
              <h4>📁 Thông Tin Dataset</h4>
              <div className="info-grid">
                <div><strong>Dataset:</strong> {results.dataset_name}</div>
                <div><strong>Total Samples:</strong> {results.n_samples.toLocaleString()}</div>
                <div><strong>Original Features:</strong> {results.n_original_features}</div>
                <div><strong>After Preprocessing:</strong> {results.n_features}</div>
                <div><strong>Categorical Features:</strong> {results.categorical_cols_count}</div>
                <div><strong>Numeric Features:</strong> {results.numeric_cols_count}</div>
                <div><strong>Train Samples (80%):</strong> {results.n_train_samples}</div>
                <div><strong>Test Samples (20%):</strong> {results.n_test_samples}</div>
              </div>
            </div>

            {/* Comparison Results */}
            <div className="comparison-results">
              <h4>⚡ Kết Quả So Sánh Chính</h4>
              
              <div className="result-item">
                <div className="label">Baseline SVM (RBF Kernel)</div>
                <div className="value baseline">
                  {results.baseline_accuracy !== null && typeof results.baseline_accuracy === 'number'
                    ? (results.baseline_accuracy * 100).toFixed(2)
                    : 'N/A'}%
                </div>
              </div>

              <div className="result-item">
                <div className="label">EBM-SVM Ensemble (4 Models)</div>
                <div className="value ensemble">
                  {results.optimized_accuracy !== null && typeof results.optimized_accuracy === 'number'
                    ? (results.optimized_accuracy * 100).toFixed(2)
                    : 'N/A'}%
                </div>
              </div>

              <div className={`result-item improvement ${results.improvement_pct > 0 ? 'positive' : 'neutral'}`}>
                <div className="label">Cải Thiện</div>
                <div className="value">
                  {results.improvement_pct !== null && typeof results.improvement_pct === 'number'
                    ? (results.improvement_pct > 0 ? '+' : '') + results.improvement_pct.toFixed(2)
                    : 'N/A'}%
                </div>
              </div>
            </div>

            {/* Models Configuration */}
            {results.baseline_config && results.ensemble_config && (
              <div className="models-config">
                <h4>⚙️ Cấu Hình Mô Hình</h4>
                
                <div className="config-item">
                  <h5>Baseline Model: {results.baseline_config.algorithm}</h5>
                  <ul>
                    <li>Kernel: {results.baseline_config.kernel}</li>
                    <li>C (Regularization): {results.baseline_config.C}</li>
                    <li>Gamma: {results.baseline_config.gamma}</li>
                    <li>Features: {results.baseline_config.features_used}</li>
                  </ul>
                </div>

                <div className="config-item">
                  <h5>Ensemble Model: {results.ensemble_config.algorithm}</h5>
                  <ul>
                    <li>Voting Strategy: {results.ensemble_config.voting}</li>
                    <li>Number of Models: {results.ensemble_config.models.length}</li>
                    <li>Models:
                      <ul>
                        {results.ensemble_config.models.map((model, idx) => (
                          <li key={idx}>{model.name}: {model.kernel} kernel, C={model.C}</li>
                        ))}
                      </ul>
                    </li>
                    <li>Features: {results.ensemble_config.features_used}</li>
                  </ul>
                </div>
              </div>
            )}

            {/* Detailed Results */}
            {results.detailed_results && (
              <div className="detailed-results">
                <h4>📋 Chi Tiết Kết Quả</h4>
                
                {results.detailed_results.preprocessing && (
                  <div className="detail-section">
                    <h5>Tiền Xử Lý Dữ Liệu (Preprocessing):</h5>
                    <ul>
                      <li>Xử lý giá trị thiếu: ? → "Unknown", NaN → median</li>
                      <li>Encoding categorical: One-hot encoding (drop_first=True)</li>
                      <li>Chuẩn hóa: StandardScaler trên tất cả features</li>
                      <li>Chia dữ liệu: 80% train, 20% test (stratified)</li>
                    </ul>
                  </div>
                )}

                <div className="detail-section">
                  <h5>Kết Quả Cuối Cùng:</h5>
                  <ul>
                    <li>Baseline Accuracy: {results.baseline_accuracy ? (results.baseline_accuracy * 100).toFixed(2) + '%' : 'N/A'}</li>
                    <li>Ensemble Accuracy: {results.optimized_accuracy ? (results.optimized_accuracy * 100).toFixed(2) + '%' : 'N/A'}</li>
                    <li>Improvement: {results.improvement_pct ? (results.improvement_pct > 0 ? '+' : '') + results.improvement_pct.toFixed(2) + '%' : 'N/A'}</li>
                    <li>Best Model: {results.best_model}</li>
                  </ul>
                </div>
              </div>
            )}

            {/* Methodology (Collapsible) */}
            {results.methodology && (
              <details className="methodology-section">
                <summary><strong>📖 Phương Pháp Chi Tiết (Thesis Methodology)</strong></summary>
                <pre className="methodology-text">{results.methodology}</pre>
              </details>
            )}

            {/* Timing */}
            <div className="timing-info">
              <h4>⏱️ Thời Gian Thực Thi</h4>
              <p><strong>Total: {results.total_time !== null && typeof results.total_time === 'number' ? results.total_time.toFixed(2) : 'N/A'}s</strong></p>
            </div>

            {/* Success/Failure Message */}
            {results.improvement_pct > 0 ? (
              <div className="success-message">
                ✓ THÀNH CÔNG! EBM-SVM cải thiện {results.improvement_pct.toFixed(2)}% so với Baseline SVM
              </div>
            ) : (
              <div className="warning-message">
                ⚠ Kết quả bằng hoặc tương đương baseline
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default UniversalPipeline;
