import React, { useState, useEffect } from 'react';
import './App.css';
import UniversalPipeline from './UniversalPipeline';

const API_BASE = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [textData, setTextData] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendUp, setBackendUp] = useState(false);
  const [uploadedFileId, setUploadedFileId] = useState(null);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [history, setHistory] = useState([]);
  const [selectedHistory, setSelectedHistory] = useState(null);

  // Check backend health on mount and every 5 seconds
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        setBackendUp(response.ok);
      } catch {
        setBackendUp(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => clearInterval(interval);
  }, []);

  // Load history on mount
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
    } catch {
      console.error('Failed to load history');
    }
  };

  const saveExecutionToHistory = async (algoName, fileId, config, resultsData, execTime) => {
    try {
      // Generate unique execution ID
      const executionId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      const response = await fetch(`${API_BASE}/api/history/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          execution_id: executionId,
          algorithm_name: algoName,
          file_id: fileId,
          parameters: config || {},
          results: resultsData,
          execution_time: execTime,
          status: 'success'
        }),
      });
      if (response.ok) {
        const data = await response.json();
        await loadHistory();
        console.log('Execution saved to history:', executionId);
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
      options = { method: 'POST', body: formData };
    } else {
      options = {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'text=' + encodeURIComponent(textData),
      };
    }

    if (!backendUp) {
      alert('Backend is unreachable');
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
      if (response.ok && data.file_id) {
        setUploadedFileId(data.file_id);
        alert('✅ Data uploaded successfully!');
      } else {
        alert('❌ Upload error: ' + (data.detail || 'Unknown error'));
      }
    } catch (error) {
      console.error('Upload exception:', error);
      alert('Upload failed: ' + error.message);
    }
    setLoading(false);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString('vi-VN');
  };

  const viewHistoryItem = (item) => {
    setSelectedHistory(item);
    setResults(item.results);
    setActiveTab('results');
  };

  const deleteHistoryItem = async (executionId) => {
    if (window.confirm('Xóa lịch sử này?')) {
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
        console.error('Failed to delete:', error);
      }
    }
  };

  return (
    <div className="App">
      {/* HEADER */}
      <header className="thesis-header">
        <div className="header-content">
          <div className="logo-section">
            <h1>🎓 EBM-SVM Ensemble</h1>
            <p className="subtitle">Phương Pháp Học Máy Tập Hợp dựa trên Máy Vector Hỗ Trợ</p>
          </div>
          <div className="thesis-info">
            <p><strong>Khóa Luận Tốt Nghiệp Đại Học</strong></p>
            <p>Phân Tích So Sánh: SVM vs EBM-SVM Ensemble</p>
          </div>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="main-content">
        {/* Intro Section */}
        <section className="intro-section">
          <div className="container">
            <h2>📚 Tổng Quan Đề Tài</h2>
            <div className="intro-grid">
              <div className="intro-card">
                <h3>🎯 Mục Tiêu</h3>
                <p>Phát triển và xác thực một phương pháp Ensemble-based SVM cho thấy hiệu suất phân loại vượt trội so với SVM cơ bản trên nhiều bộ dữ liệu khác nhau.</p>
              </div>
              <div className="intro-card">
                <h3>🔬 Phương Pháp</h3>
                <p>Ensemble SVM đa-kernel sử dụng soft voting (RBF, Polynomial, Linear kernels) với cơ chế fallback tự động để đảm bảo hiệu suất ≥ baseline.</p>
              </div>
              <div className="intro-card">
                <h3>📊 Bộ Dữ Liệu</h3>
                <p>Đánh giá trên Adult, Wine, Iris, và Breast Cancer datasets với xử lý dữ liệu toàn diện và kỹ thuật trích xuất đặc trưng.</p>
              </div>
            </div>
          </div>
        </section>

        {/* TABS */}
        <div className="tabs-container">
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              📤 Tải Lên & So Sánh
            </button>
            <button
              className={`tab ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              📜 Lịch Sử ({history.length})
            </button>
            <button
              className={`tab ${activeTab === 'results' ? 'active' : ''}`}
              onClick={() => setActiveTab('results')}
            >
              📊 Kết Quả
            </button>
          </div>

          {/* UPLOAD TAB */}
          {activeTab === 'upload' && (
            <div className="tab-content">
              <div className="container">
                <h2>Tải Lên Dữ Liệu CSV</h2>
                
                <div className={`status ${backendUp ? 'online' : 'offline'}`}>
                  Trạng Thái Backend: {backendUp ? '✅ Online' : '❌ Offline'}
                </div>

                <div style={{ marginTop: '20px' }}>
                  <label style={{ fontWeight: 'bold', marginBottom: '10px', display: 'block' }}>
                    Chọn File CSV:
                  </label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setFile(e.target.files[0])}
                  />
                  {file && <p style={{ marginTop: '10px', color: '#2e7d32' }}>✓ File đã chọn: {file.name}</p>}
                </div>

                <div style={{ marginTop: '20px', marginBottom: '20px' }}>
                  <label style={{ fontWeight: 'bold', marginBottom: '10px', display: 'block' }}>
                    Hoặc dán dữ liệu CSV:
                  </label>
                  <textarea
                    value={textData}
                    onChange={(e) => setTextData(e.target.value)}
                    placeholder="Dán dữ liệu CSV tại đây..."
                    style={{
                      width: '100%',
                      height: '150px',
                      padding: '10px',
                      borderRadius: '6px',
                      border: '1px solid #0d47a1',
                      fontFamily: 'monospace',
                      fontSize: '12px',
                    }}
                  />
                </div>

                <button
                  onClick={handleUpload}
                  disabled={loading || !backendUp || (!file && !textData.trim())}
                  style={{
                    padding: '14px 32px',
                    fontSize: '1.05em',
                    width: '200px',
                  }}
                >
                  {loading ? '⏳ Đang tải lên...' : '📤 Tải Lên Dữ Liệu'}
                </button>

                {uploadedFileId && (
                  <div className="success-message" style={{ marginTop: '20px' }}>
                    ✅ Dữ liệu đã tải lên thành công! File ID: <code>{uploadedFileId}</code>
                  </div>
                )}
              </div>

              {/* Universal Pipeline Component */}
              {uploadedFileId && (
                <UniversalPipeline
                  fileId={uploadedFileId}
                  onResults={(data) => {
                    setResults(data);
                    setActiveTab('results');
                    const startTime = Date.now();
                    saveExecutionToHistory(
                      'SVM vs EBM-SVM Comparison',
                      uploadedFileId,
                      { fileId: uploadedFileId },
                      data,
                      (Date.now() - startTime) / 1000
                    );
                  }}
                />
              )}
            </div>
          )}

          {/* HISTORY TAB */}
          {activeTab === 'history' && (
            <div className="tab-content">
              <div className="container">
                <h3>📜 Lịch Sử Thực Thi ({history.length})</h3>

                {history.length === 0 ? (
                  <div className="warning-message">
                    Chưa có lịch sử thực thi. Tải dữ liệu lên và chạy so sánh để bắt đầu!
                  </div>
                ) : (
                  <div className="history-list">
                    {history.map((item) => (
                      <div key={item.execution_id} className="history-item">
                        <div className="history-item-header">
                          <div>
                            <div className="history-item-title">{item.algorithm_name}</div>
                            <div className="history-item-date">{formatDate(item.timestamp)}</div>
                            <div className="history-item-meta">
                              File ID: {item.file_id} | Thời Gian: {item.execution_time?.toFixed(2)}s
                            </div>
                          </div>
                        </div>
                        <div className="history-button-group">
                          <button
                            onClick={() => viewHistoryItem(item)}
                            style={{
                              background: 'linear-gradient(135deg, #0d47a1 0%, #1a237e 100%)',
                            }}
                          >
                            Xem Kết Quả
                          </button>
                          <button
                            onClick={() => deleteHistoryItem(item.execution_id)}
                            className="delete-button"
                          >
                            Xóa
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* RESULTS TAB */}
          {activeTab === 'results' && results && (
            <div className="tab-content">
              <div className="results-section">
                <h3>📊 Kết Quả So Sánh</h3>

                {selectedHistory && (
                  <div className="info-grid" style={{ marginBottom: '30px' }}>
                    <div>
                      <strong>Thuật Toán:</strong> {selectedHistory.algorithm_name}
                    </div>
                    <div>
                      <strong>Thời Gian Thực Thi:</strong> {selectedHistory.execution_time?.toFixed(2)}s
                    </div>
                    <div>
                      <strong>Ngày Giờ:</strong> {formatDate(selectedHistory.timestamp)}
                    </div>
                    <div>
                      <strong>File ID:</strong> {selectedHistory.file_id}
                    </div>
                  </div>
                )}

                {/* Display results using UniversalPipeline display logic */}
                <UniversalPipeline
                  fileId={null}
                  displayResults={results}
                  onResults={() => {}}
                />
              </div>
            </div>
          )}

          {activeTab === 'results' && !results && (
            <div className="container">
              <div className="warning-message" style={{ textAlign: 'center' }}>
                Không có kết quả để hiển thị. Tải dữ liệu và chạy so sánh trước!
              </div>
            </div>
          )}
        </div>
      </main>

      {/* FOOTER */}
      <footer>
        <p>&copy; 2026 Khóa Luận Tốt Nghiệp EBM-SVM | Phương Pháp Học Máy Tập Hợp dựa trên Máy Vector Hỗ Trợ</p>
      </footer>
    </div>
  );
}

export default App;
