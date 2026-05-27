import React, { useState, useEffect } from 'react';
import './App.css';
import UniversalPipeline from './UniversalPipeline';

function App() {
  const [file, setFile] = useState(null);
  const [textData, setTextData] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendUp, setBackendUp] = useState(false);
  const [runningAlgo, setRunningAlgo] = useState('');
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('results');
  const [selectedHistory, setSelectedHistory] = useState(null);
  const [lastExecutionId, setLastExecutionId] = useState(null);
  const [uploadedFileId, setUploadedFileId] = useState(null);

  // Use direct backend URL (CORS enabled in backend) — more reliable than dev-server proxy
  const API_BASE = 'http://localhost:8000';

  // Periodically check backend health
  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (!mounted) return;
        setBackendUp(res.ok);
      } catch (_) {
        if (!mounted) return;
        setBackendUp(false);
      }
    };
    check();
    const id = setInterval(check, 3000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  // Load execution history on mount
  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/history/all`);
      if (response.ok) {
        const data = await response.json();
        setHistory(data.histories || []);
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const saveExecutionToHistory = async (
    algorithmName,
    fileId,
    parameters,
    results,
    executionTime
  ) => {
    try {
      const executionId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setLastExecutionId(executionId);

      // Save execution history to backend
      const historyPayload = {
        execution_id: executionId,
        algorithm_name: algorithmName,
        file_id: fileId || 'uploaded_data',
        parameters: parameters,
        results: results,
        execution_time: executionTime,
        analysis: null,
        status: 'success'
      };

      const response = await fetch(`${API_BASE}/api/history/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(historyPayload)
      });

      if (response.ok) {
        const savedData = await response.json();
        
        // Trigger automatic analysis if not already done
        try {
          await fetch(`${API_BASE}/api/history/analyze?execution_id=${executionId}`, {
            method: 'POST',
          });
        } catch (e) {
          console.error('Failed to analyze results:', e);
        }
        
        // Reload history to show new entry
        await loadHistory();
        console.log('Execution saved to history:', executionId);
      } else {
        console.error('Failed to save history - server error');
      }
    } catch (error) {
      console.error('Failed to save history:', error);
    }
  };

  const handleUpload = async () => {
    if (!file && !textData.trim()) return;
    setLoading(true);
    let options = {};
    if (file) {
      let formData = new FormData();
      formData.append('file', file);
      options = {
        method: 'POST',
        body: formData,
      };
    } else {
      options = {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'text=' + encodeURIComponent(textData),
      };
    }

    if (!backendUp) {
      alert('Backend is unreachable — check server and try again');
      setLoading(false);
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/files/upload`, options);
      const text = await response.text();
      let data = {};
      try {
        data = JSON.parse(text);
      } catch (_) {
        data = { detail: text };
      }
      if (response.ok) {
        // Extract and store file_id from response
        if (data.file_id) {
          setUploadedFileId(data.file_id);
          console.log('File uploaded with ID:', data.file_id);
        }
        alert('Data uploaded successfully!');
      } else {
        alert(data.detail || 'Upload error');
      }
    } catch (error) {
      console.error('Upload exception:', error);
      alert('Upload failed: ' + (error.message || error));
    }
    setLoading(false);
  };

  const handleRunAll = async () => {
    setLoading(true);
    const startTime = Date.now();
    try {
      const [svmRes, ebmRes] = await Promise.all([
        fetch(`${API_BASE}/run-svm`, { method: 'POST' }),
        fetch(`${API_BASE}/run-ebm-svm`, { method: 'POST' }),
      ]);
      const svmData = await svmRes.json();
      const ebmData = await ebmRes.json();
      const executionTime = (Date.now() - startTime) / 1000;
      
      const resultsData = { svm: svmData, ebm: ebmData };
      setResults(resultsData);

      // Save to history
      await saveExecutionToHistory(
        'SVM & EBM-SVM',
        'uploaded_data',
        { algorithm: 'both' },
        resultsData,
        executionTime
      );

      setActiveTab('results');
    } catch (error) {
      alert('Run failed: ' + error.message);
    }
    setLoading(false);
  };

  const runAlgorithm = async (endpoint) => {
    setRunningAlgo(endpoint);
    setLoading(true);
    const startTime = Date.now();
    try {
      const response = await fetch(`${API_BASE}/${endpoint}`, {
        method: 'POST',
      });
      const data = await response.json();
      const executionTime = (Date.now() - startTime) / 1000;
      
      if (response.ok) {
        setResults(data);

        // Save to history
        await saveExecutionToHistory(
          endpoint.replace('run-', '').replace('-', ' ').toUpperCase(),
          'uploaded_data',
          { algorithm: endpoint },
          data,
          executionTime
        );

        setActiveTab('results');
      } else {
        alert(data.detail);
      }
    } catch (error) {
      alert('Run failed: ' + error.message);
    }
    setLoading(false);
    setRunningAlgo('');
  };

  const viewHistoryItem = (item) => {
    setSelectedHistory(item);
    setResults(item.results);
    setActiveTab('analysis');
  };

  const deleteHistoryItem = async (executionId) => {
    if (window.confirm('Bạn có chắc muốn xóa lịch sử này?')) {
      try {
        const response = await fetch(`${API_BASE}/api/history/${executionId}`, {
          method: 'DELETE',
        });
        if (response.ok) {
          await loadHistory();
          if (selectedHistory?.execution_id === executionId) {
            setSelectedHistory(null);
          }
        }
      } catch (error) {
        console.error('Failed to delete history:', error);
      }
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString('vi-VN');
  };

  return (
    <div className="App">
      <header className="thesis-header">
        <div className="header-content">
          <div className="logo-section">
            <h1>🎓 EBM-SVM Ensemble</h1>
            <p className="subtitle">Ensemble-Based Machine Learning with Support Vector Machines</p>
          </div>
          <div className="thesis-info">
            <p><strong>Capstone Project / Khóa Luận Tốt Nghiệp</strong></p>
            <p>Comparative Analysis: SVM vs EBM-SVM Ensemble</p>
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="intro-section">
          <div className="container">
            <h2>📚 Project Overview</h2>
            <div className="intro-grid">
              <div className="intro-card">
                <h3>🎯 Objective</h3>
                <p>Develop and validate an ensemble-based SVM approach that demonstrates superior classification performance compared to standard SVM baseline across multiple datasets.</p>
              </div>
              <div className="intro-card">
                <h3>🔬 Methodology</h3>
                <p>Multi-kernel SVM ensemble using soft voting (RBF, Polynomial, Linear kernels) with automatic fallback mechanism to guarantee performance ≥ baseline.</p>
              </div>
              <div className="intro-card">
                <h3>📊 Datasets</h3>
                <p>Evaluation on Adult, Wine, Iris, and Breast Cancer datasets with comprehensive preprocessing and feature engineering.</p>
              </div>
            </div>
          </div>
        </section>

      <div className="upload-section">
        <h2>Upload Data & Run Algorithms</h2>
        <div className={`status ${backendUp ? 'online' : 'offline'}`}>
          Backend Status: {backendUp ? 'Online' : 'Offline'}
        </div>
        <div>
          <label>Upload CSV File:</label>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </div>
        <br />
        <button onClick={handleUpload} disabled={loading || !backendUp || !file}>
          {loading ? 'Uploading...' : 'Upload Data'}
        </button>
      </div>

      {/* Universal Pipeline Component */}
      <UniversalPipeline fileId={uploadedFileId} onResults={(data) => {
        setResults(data);
        setActiveTab('results');
      }} />

      <div className="tabs-container">
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
          >
            Kết quả ({results ? '1' : '0'})
          </button>
          <button
            className={`tab ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            Lịch sử ({history.length})
          </button>
          <button
            className={`tab ${activeTab === 'analysis' ? 'active' : ''}`}
            onClick={() => setActiveTab('analysis')}
            disabled={!selectedHistory}
          >
            Phân tích
          </button>
        </div>

        <div className="tab-content">
          {/* Results Tab */}
          {activeTab === 'results' && results && (
            <div className="results">
              {results.svm && results.ebm ? (
                <>
                  <h3>So sánh kết quả SVM vs EBM-SVM</h3>
                  <div className="metric">SVM Accuracy: {(results.svm.accuracy * 100).toFixed(2)}%</div>
                  <div className="metric">EBM-SVM Accuracy: {(results.ebm.accuracy * 100).toFixed(2)}%</div>
                  {results.svm.report && (
                    <div>
                      <h4>SVM Classification Report</h4>
                      <div className="report">{JSON.stringify(results.svm.report, null, 2)}</div>
                    </div>
                  )}
                  {results.svm.confusion_matrix && (
                    <div className="confusion-matrix">
                      <h4>SVM Confusion Matrix</h4>
                      <table>
                        <thead>
                          <tr>
                            <th>Predicted \ Actual</th>
                            {results.svm.confusion_matrix[0].map((_, i) => <th key={i}>Class {i}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {results.svm.confusion_matrix.map((row, i) => (
                            <tr key={i}>
                              <th>Class {i}</th>
                              {row.map((cell, j) => (
                                <td key={j}>{cell}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {results.ebm.report && (
                    <div>
                      <h4>EBM-SVM Classification Report</h4>
                      <div className="report">{JSON.stringify(results.ebm.report, null, 2)}</div>
                    </div>
                  )}
                  {results.ebm.confusion_matrix && (
                    <div className="confusion-matrix">
                      <h4>EBM-SVM Confusion Matrix</h4>
                      <table>
                        <thead>
                          <tr>
                            <th>Predicted \ Actual</th>
                            {results.ebm.confusion_matrix[0].map((_, i) => <th key={i}>Class {i}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {results.ebm.confusion_matrix.map((row, i) => (
                            <tr key={i}>
                              <th>Class {i}</th>
                              {row.map((cell, j) => (
                                <td key={j}>{cell}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <h3>Kết quả</h3>
                  {results.accuracy && <div className="metric">Accuracy: {(results.accuracy * 100).toFixed(2)}%</div>}
                  {results.mean_accuracy && <div className="metric">Mean Accuracy (CV): {(results.mean_accuracy * 100).toFixed(2)}% ± {(results.std_accuracy * 100).toFixed(2)}%</div>}
                  {results.report && (
                    <div>
                      <h4>Classification Report</h4>
                      <div className="report">{JSON.stringify(results.report, null, 2)}</div>
                    </div>
                  )}
                  {results.confusion_matrix && (
                    <div className="confusion-matrix">
                      <h4>Confusion Matrix</h4>
                      <table>
                        <thead>
                          <tr>
                            <th>Predicted \ Actual</th>
                            {results.confusion_matrix[0].map((_, i) => <th key={i}>Class {i}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {results.confusion_matrix.map((row, i) => (
                            <tr key={i}>
                              <th>Class {i}</th>
                              {row.map((cell, j) => (
                                <td key={j}>{cell}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {results.scores && <div className="cv-scores">CV Scores: {results.scores.map(s => (s * 100).toFixed(2) + '%').join(', ')}</div>}
                </>
              )}
            </div>
          )}

          {activeTab === 'results' && !results && (
            <div className="empty-state">
              <p>Chạy một giải thuật để xem kết quả tại đây</p>
            </div>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="history-container">
              <h3>Lịch sử Thực thi ({history.length})</h3>
              {history.length === 0 ? (
                <div className="empty-state">
                  <p>Chưa có lịch sử thực thi</p>
                </div>
              ) : (
                <div className="history-list">
                  {history.map((item, index) => (
                    <div
                      key={item.execution_id}
                      className={`history-item ${selectedHistory?.execution_id === item.execution_id ? 'selected' : ''}`}
                    >
                      <div className="history-item-header">
                        <div className="history-item-info">
                          <strong>{item.algorithm_name}</strong>
                          <span className="timestamp">{formatDate(item.timestamp)}</span>
                        </div>
                        <div className="history-item-actions">
                          <button
                            onClick={() => viewHistoryItem(item)}
                            className="btn-view"
                          >
                            Xem
                          </button>
                          <button
                            onClick={() => deleteHistoryItem(item.execution_id)}
                            className="btn-delete"
                          >
                            Xóa
                          </button>
                        </div>
                      </div>
                      {selectedHistory?.execution_id === item.execution_id && (
                        <div className="history-item-details">
                          <p>Execution Time: {item.execution_time.toFixed(2)}s</p>
                          <p>File ID: {item.file_id}</p>
                          {item.analysis && (
                            <div className="analysis-preview">
                              <p><strong>Accuracy:</strong> {(item.analysis.accuracy_score * 100).toFixed(2)}%</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Analysis Tab */}
          {activeTab === 'analysis' && selectedHistory && (
            <div className="analysis-container">
              <h3>Phân tích Kết quả - {selectedHistory.algorithm_name}</h3>
              <div className="analysis-info">
                <p><strong>Thời gian thực thi:</strong> {selectedHistory.execution_time.toFixed(2)}s</p>
                <p><strong>Thời gian chạy:</strong> {formatDate(selectedHistory.timestamp)}</p>
              </div>

              {selectedHistory.analysis ? (
                <div className="analysis-results">
                  <div className="analysis-section">
                    <h4>📊 Hiệu suất</h4>
                    <div className="metric analysis-metric">
                      Accuracy: {(selectedHistory.analysis.accuracy_score * 100).toFixed(2)}%
                    </div>
                    {selectedHistory.analysis.best_metric && (
                      <div className="metric">
                        <strong>Best Metric:</strong> {selectedHistory.analysis.best_metric}
                      </div>
                    )}
                  </div>

                  <div className="analysis-section">
                    <h4>🔍 Phát hiện chính</h4>
                    <ul className="findings-list">
                      {selectedHistory.analysis.key_findings.map((finding, idx) => (
                        <li key={idx}>{finding}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="analysis-section">
                    <h4>💡 Khuyến nghị</h4>
                    <ul className="recommendations-list">
                      {selectedHistory.analysis.recommendations.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <p>Phân tích chưa có sẵn cho kết quả này</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
