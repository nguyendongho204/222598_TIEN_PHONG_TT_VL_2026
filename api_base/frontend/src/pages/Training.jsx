import { useState, useRef, useCallback, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useTranslation } from 'react-i18next'
import { trainDataset } from '../services/api'

const DEFAULT_PARAMS = {
  test_size: 0.2,
  num_runs: 5,
}

const PROGRESS_PHASES_KEY = [
  'phase_load', 'phase_preprocess', 'phase_pca', 'phase_jepa', 'phase_svm', 'phase_eval',
]

const TRAINING_STORAGE_KEY = 'trainingResult'

const STEPS = ['upload', 'split', 'execute', 'cv', 'summary']

export default function Training() {
  const { t } = useTranslation()
  const PROGRESS_PHASES = PROGRESS_PHASES_KEY.map((key, i) => ({
    pct: [10, 25, 40, 55, 70, 85][i],
    msg: t(`training.${key}`),
  }))
  const fileInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState({ pct: 0, msg: '' })
  const [currentStep, setCurrentStep] = useState(0)
  const [csvPreview, setCsvPreview] = useState(null)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(TRAINING_STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored)
        if (parsed.status === 'complete') {
          setResult(parsed.result)
          setCurrentStep(3)
        }
      }
    } catch {}
  }, [])

  const handleFileChange = useCallback((e) => {
    const f = e.target.files[0]
    if (f) {
      setFile(f)
      previewCsv(f)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.name.endsWith('.csv')) {
      setFile(f)
      previewCsv(f)
    }
  }, [])

  const previewCsv = (f) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target.result
      const lines = text.split('\n').filter(l => l.trim())
      if (lines.length > 0) {
        const headers = lines[0].split(',').map(h => h.trim())
        const dataRows = lines.slice(1)
        setCsvPreview({
          fileName: f.name,
          fileSize: f.size,
          rows: dataRows.length,
          cols: headers.length,
          headers: headers,
          preview: dataRows.slice(0, 3).map(row => row.split(',').map(c => c.trim())),
        })
      }
    }
    reader.readAsText(f)
  }

  const handleParamChange = useCallback((key, value) => {
    setParams((prev) => ({ ...prev, [key]: parseFloat(value) || value }))
  }, [])

  const handleTrain = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setCurrentStep(2)
    setProgress({ pct: 5, msg: t('training.phase_start') || 'Starting...' })

    localStorage.setItem(TRAINING_STORAGE_KEY, JSON.stringify({
      status: 'running',
      fileName: file.name,
      timestamp: new Date().toISOString(),
    }))

    const timers = PROGRESS_PHASES.map((phase, i) =>
      setTimeout(() => {
        setProgress(phase)
      }, (i + 1) * 4000)
    )

    try {
      const res = await trainDataset(file, params)
      setProgress({ pct: 100, msg: t('training.phase_complete') || 'Complete!' })
      localStorage.setItem(TRAINING_STORAGE_KEY, JSON.stringify({
        status: 'complete',
        result: res,
        timestamp: new Date().toISOString(),
      }))
      setTimeout(() => {
        setResult(res)
        setCurrentStep(3)
      }, 300)
    } catch (err) {
      setError(err.message)
    } finally {
      timers.forEach(clearTimeout)
      setTimeout(() => {
        setProgress({ pct: 0, msg: '' })
        setLoading(false)
      }, 500)
    }
  }

  const canNext = () => {
    if (currentStep === 0) return !!file
    if (currentStep === 1) return true
    return false
  }

  const stepLabels = [
    t('training.step_upload') || 'Upload',
    t('training.step_split') || 'Split',
    t('training.step_execute') || 'Execute',
    t('training.step_cv') || 'K-Fold CV',
    t('training.step_summary') || 'Summary',
  ]

  const chartData = result
    ? [
        { name: 'JEPA+SVM', accuracy: +(result.accuracy * 100).toFixed(2) },
      ]
    : []

  const renderStepIndicator = () => (
    <div style={{
      display: 'flex', justifyContent: 'center', gap: '0.5rem',
      marginBottom: '2rem', flexWrap: 'wrap',
    }}>
      {STEPS.map((step, idx) => {
        const isActive = currentStep === idx
        const isDone = currentStep > idx
        return (
          <div
            key={step}
            onClick={() => {
              if (idx < currentStep) setCurrentStep(idx)
            }}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              padding: '0.6rem 1.2rem', borderRadius: '30px',
              cursor: idx < currentStep ? 'pointer' : isActive ? 'default' : 'not-allowed',
              background: isActive ? 'var(--primary)' : isDone ? 'var(--success-light)' : 'var(--bg-card-alt)',
              color: isActive ? '#fff' : isDone ? '#2e7d32' : '#999',
              fontWeight: isActive ? 700 : 400,
              fontSize: '0.85rem',
              transition: 'all 0.3s',
              border: isActive ? 'none' : '2px solid ' + (isDone ? '#a5d6a7' : '#e0e0e0'),
              opacity: isActive || isDone ? 1 : 0.7,
            }}
          >
            <span style={{
              width: 22, height: 22, borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.7rem', fontWeight: 700,
              background: isActive ? 'rgba(255,255,255,0.3)' : isDone ? '#a5d6a7' : '#ddd',
              color: isActive ? '#fff' : isDone ? '#fff' : '#999',
            }}>{idx + 1}</span>
            <span>{stepLabels[idx]}</span>
          </div>
        )
      })}
    </div>
  )

  const renderStep0 = () => (
    <div className="card">
      <h2>{t('training.title_upload')}</h2>
      <div
        className="drop-zone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: '2px dashed var(--primary)',
          borderRadius: 12,
          padding: '2rem',
          textAlign: 'center',
          cursor: 'pointer',
          background: file ? '#f0fff0' : '#f8f9ff',
          transition: 'all 0.2s',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        {file ? (
          <div>
            <p style={{ fontSize: '1.1rem', fontWeight: 600, color: '#28a745' }}>{file.name}</p>
            <p style={{ color: '#888', fontSize: '0.85rem', marginTop: '0.3rem' }}>
              {(file.size / 1024).toFixed(1)} {t('training.kb')} — {t('training.click_change')}
            </p>
          </div>
        ) : (
          <div>
            <p style={{ fontSize: '2rem', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>+</p>
            <p style={{ fontWeight: 600, color: 'var(--primary)' }}>{t('training.drop_text')}</p>
            <p style={{ color: '#888', fontSize: '0.85rem', marginTop: '0.3rem' }}>{t('training.click_select')}</p>
          </div>
        )}
      </div>

      {csvPreview && (
        <div style={{ marginTop: '1.5rem' }}>
          <h3 style={{ fontSize: '0.95rem', color: '#555', marginBottom: '0.5rem' }}>
            Dataset Preview
          </h3>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
            <div className="metric-card" style={{ flex: '1 1 80px', padding: '0.6rem' }}>
              <div className="metric-label">Rows</div>
              <div className="metric-value" style={{ fontSize: '1rem' }}>{csvPreview.rows}</div>
            </div>
            <div className="metric-card" style={{ flex: '1 1 80px', padding: '0.6rem' }}>
              <div className="metric-label">Columns</div>
              <div className="metric-value" style={{ fontSize: '1rem' }}>{csvPreview.cols}</div>
            </div>
            <div className="metric-card" style={{ flex: '1 1 80px', padding: '0.6rem' }}>
              <div className="metric-label">Size</div>
              <div className="metric-value" style={{ fontSize: '1rem' }}>{(csvPreview.fileSize / 1024).toFixed(1)} KB</div>
            </div>
          </div>
          <div style={{ overflowX: 'auto', fontSize: '0.8rem' }}>
            <table className="metrics-table">
              <thead>
                <tr>
                  {csvPreview.headers.map((h, i) => (
                    <th key={i} style={{ padding: '0.4rem 0.6rem' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {csvPreview.preview.map((row, ri) => (
                  <tr key={ri}>
                    {row.map((cell, ci) => (
                      <td key={ci} style={{ padding: '0.4rem 0.6rem' }}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ color: '#888', fontSize: '0.75rem', marginTop: '0.3rem' }}>
            Showing first 3 rows — last column is treated as label
          </p>
        </div>
      )}
    </div>
  )

  const renderStep1 = () => (
    <div className="card">
      <h2>{t('training.title_split')}</h2>
      <p style={{ color: '#888', fontSize: '0.85rem', marginBottom: '1rem' }}>
        {t('training.split_desc')}
      </p>
      <div className="form-row">
        <div className="form-group">
          <label>{t('training.test_size')} ({(params.test_size * 100).toFixed(0)}% / {((1 - params.test_size) * 100).toFixed(0)}%)</label>
          <input
            type="range"
            min="0.05"
            max="0.5"
            step="0.05"
            value={params.test_size}
            onChange={(e) => handleParamChange('test_size', e.target.value)}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#999', marginTop: '0.2rem' }}>
            <span>Train: {((1 - params.test_size) * 100).toFixed(0)}%</span>
            <span>Test: {(params.test_size * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      <div className="form-row" style={{ marginTop: '0.5rem' }}>
        <div className="form-group">
          <label>{t('training.num_folds')} ({params.num_runs})</label>
          <input
            type="range"
            min="2"
            max="10"
            step="1"
            value={params.num_runs}
            onChange={(e) => handleParamChange('num_runs', parseInt(e.target.value))}
          />
          <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.2rem' }}>
            {t('training.num_folds_desc')}
          </div>
        </div>
      </div>

      {csvPreview && (
        <div style={{
          background: '#f8f9ff', borderRadius: 10, padding: '1rem', marginTop: '1rem',
          border: '1px solid #e8e8ff',
        }}>
          <p style={{ fontSize: '0.85rem', color: 'var(--primary)', fontWeight: 600, marginBottom: '0.5rem' }}>
            📋 {t('training.split_summary')}
          </p>
          <p style={{ fontSize: '0.8rem', color: '#666' }}>
            {t('training.dataset')}: <strong>{csvPreview.fileName}</strong> |{' '}
            {t('training.samples')}: <strong>{csvPreview.rows}</strong> |{' '}
            {t('training.features')}: <strong>{csvPreview.cols - 1}</strong> |{' '}
            Train: <strong>{Math.round(csvPreview.rows * (1 - params.test_size))}</strong> |{' '}
            Test: <strong>{Math.round(csvPreview.rows * params.test_size)}</strong>
          </p>
        </div>
      )}
    </div>
  )

  const renderStep2 = () => (
    <div className="card">
      <h2>{t('training.title_execute')}</h2>
      <p style={{ color: '#888', fontSize: '0.85rem', marginBottom: '1rem' }}>
        {t('training.execute_desc')}
      </p>

      {!loading && !result && (
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <button
            className="btn btn-primary"
            onClick={handleTrain}
            style={{ padding: '0.8rem 2.5rem', fontSize: '1rem' }}
          >
            {t('training.start_train')}
          </button>
        </div>
      )}

      {loading && (
        <div>
          <div style={{ marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--primary)', fontWeight: 600 }}>{progress.msg}</span>
            <span style={{ color: '#888' }}>{progress.pct}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress.pct}%` }} />
          </div>
          <div className="progress-phases">
            {PROGRESS_PHASES.map((phase, i) => (
              <div key={i} className={`phase-dot ${progress.pct >= phase.pct ? 'done' : ''}`} />
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="status-bar status-error">{error}</div>
      )}

      {result && !loading && (
        <div className="status-bar status-success">
          {t('training.complete')} — {t('training.proceed_cv')}
        </div>
      )}
    </div>
  )

  const renderStep3 = () => (
    <div className="card">
        <h2>{t('training.title_cv')}</h2>
      <p style={{ color: '#888', fontSize: '0.85rem', marginBottom: '1rem' }}>
        {t('training.cv_desc')}
      </p>

      {result && result.per_fold_accuracies ? (
        <div>
          <div style={{ overflowX: 'auto', marginBottom: '1rem' }}>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>{t('training.cv_fold')}</th>
                  <th>{t('training.jepa_svm')}</th>
                </tr>
              </thead>
              <tbody>
                {result.per_fold_accuracies.map((acc, i) => (
                  <tr key={i}>
                    <td><strong>{t('training.cv_fold')} {i + 1}</strong></td>
                    <td>{(acc * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="metric-grid" style={{ marginTop: '1rem' }}>
            <div className="metric-card" style={{ border: '2px solid #28a745' }}>
              <div className="metric-label" style={{ color: '#28a745' }}>{t('training.jepa_svm')}</div>
              <div className="metric-value">{(result.accuracy * 100).toFixed(2)}%</div>
              <div style={{ fontSize: '0.75rem', color: '#888' }}>
                ± {(result.accuracy_std * 100).toFixed(2)}%
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">{t('training.cv_folds')}</div>
              <div className="metric-value">{result.per_fold_accuracies.length}</div>
            </div>
          </div>

          {result.per_fold_accuracies.length > 1 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart
                data={result.per_fold_accuracies.map((acc, i) => ({
                  fold: `${i + 1}`,
                  'JEPA+SVM': +(acc * 100).toFixed(2),
                }))}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="fold" tick={{ fontSize: 12 }} label={{ value: t('training.cv_fold'), position: 'insideBottom', offset: -5 }} />
                <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                <Tooltip />
                <Bar dataKey="JEPA+SVM" fill="#28a745" radius={[4, 4, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      ) : result ? (
        <div style={{ textAlign: 'center', padding: '2rem', color: '#999' }}>
          <p>{t('training.cv_wait')}</p>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '2rem', color: '#999' }}>
          <p>{t('training.cv_wait')}</p>
        </div>
      )}
    </div>
  )

  const renderStep4 = () => (
    <div>
      {result && (
        <>
          <div className="card">
            <h2>{t('training.title_summary')} {result.per_fold_accuracies ? `(${result.per_fold_accuracies.length} ${t('training.runs')})` : ''}</h2>
            <div className="status-bar status-success">
              {t('training.complete')}
            </div>
            <div className="metric-grid">
              <div className="metric-card">
                <div className="metric-value">{result.dataset_name}</div>
                <div className="metric-label">{t('training.dataset')}</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{result.samples}</div>
                <div className="metric-label">{t('training.samples')}</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{result.features}</div>
                <div className="metric-label">{t('training.features')}</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{result.classes}</div>
                <div className="metric-label">{t('training.classes')}</div>
              </div>
              {result.pca_components != null && (
                <div className="metric-card">
                  <div className="metric-value">{result.pca_components}</div>
                  <div className="metric-label">{t('training.pca_components')}</div>
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <h2>{t('training.comparison')}</h2>
            <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
              <div className="metric-card" style={{ display: 'inline-block', border: '2px solid #28a745' }}>
                <div className="metric-label" style={{ color: '#28a745' }}>JEPA+SVM</div>
                <div className="metric-value">{(result.accuracy * 100).toFixed(2)}%</div>
                <div style={{ fontSize: '0.75rem', color: '#888' }}>
                  ± {(result.accuracy_std * 100).toFixed(2)}%
                </div>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="name" tick={{ fontSize: 13 }} />
                <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                <Tooltip formatter={(v) => [`${v}%`, 'Accuracy']} />
                <Bar dataKey="accuracy" fill="#28a745" radius={[6, 6, 0, 0]} barSize={60} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {result.classification_report && (
            <div className="card">
              <h2>{t('training.class_report')}</h2>
              <div style={{ overflowX: 'auto' }}>
                <table className="metrics-table">
                  <thead>
                    <tr>
                      <th>{t('training.col_class')}</th>
                      <th>{t('training.col_precision')}</th>
                      <th>{t('training.col_recall')}</th>
                      <th>{t('training.col_f1')}</th>
                      <th>{t('training.col_samples')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.classification_report).map(([key, val]) => {
                      if (key === 'accuracy') return null
                      if (typeof val === 'object' && val.precision != null) {
                        return (
                          <tr key={key}>
                            <td><strong>{key}</strong></td>
                            <td>{(val.precision * 100).toFixed(2)}%</td>
                            <td>{(val.recall * 100).toFixed(2)}%</td>
                            <td>{(val['f1-score'] * 100).toFixed(2)}%</td>
                            <td>{val.support}</td>
                          </tr>
                        )
                      }
                      if (key === 'macro avg' || key === 'weighted avg') {
                        return (
                          <tr key={key} style={{ background: '#f0f0ff', fontWeight: 600 }}>
                            <td>{key === 'macro avg' ? t('training.macro_avg') : t('training.weighted_avg')}</td>
                            <td>{(val.precision * 100).toFixed(2)}%</td>
                            <td>{(val.recall * 100).toFixed(2)}%</td>
                            <td>{(val['f1-score'] * 100).toFixed(2)}%</td>
                            <td>{val.support}</td>
                          </tr>
                        )
                      }
                      return null
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {result.confusion_matrix && result.confusion_matrix.length > 0 && (
            <div className="card">
              <h2>{t('training.conf_matrix')}</h2>
              <p style={{ color: '#888', fontSize: '0.85rem', marginBottom: '0.8rem' }}>
                {t('training.conf_desc')}
              </p>
              <div style={{ overflowX: 'auto' }}>
                <table className="confusion-matrix">
                  <thead>
                    <tr>
                      <th></th>
                      {result.confusion_matrix[0].map((_, i) => (
                        <th key={i}>{t('training.pred')} {i}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.confusion_matrix.map((row, i) => (
                      <tr key={i}>
                        <td><strong>{t('training.true')} {i}</strong></td>
                        {row.map((val, j) => {
                          const maxVal = Math.max(...result.confusion_matrix.map(r => Math.max(...r)))
                          const intensity = maxVal > 0 ? (val / maxVal) : 0
                          const bg = i === j
                            ? `rgba(40, 167, 69, ${0.2 + intensity * 0.6})`
                            : `rgba(220, 53, 69, ${intensity * 0.5})`
                          return (
                            <td key={j} style={{
                              background: bg,
                              textAlign: 'center',
                              fontWeight: i === j ? 700 : 400,
                              color: intensity > 0.5 ? 'white' : '#333',
                              padding: '0.6rem 1rem',
                              minWidth: 60,
                            }}>
                              {val}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )

  const renderStepContent = () => {
    switch (currentStep) {
      case 0: return renderStep0()
      case 1: return renderStep1()
      case 2: return renderStep2()
      case 3: return renderStep3()
      case 4: return renderStep4()
      default: return null
    }
  }

  return (
    <>
      {renderStepIndicator()}

      {renderStepContent()}

      <div style={{
        display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem',
        marginBottom: '2rem',
      }}>
          <button
            className="btn"
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            style={{
              padding: '0.6rem 1.5rem', opacity: currentStep === 0 ? 0.4 : 1,
              background: '#f0f0f0', border: '1px solid #ddd', borderRadius: 8,
              cursor: currentStep === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            {t('training.btn_prev')}
          </button>

        {currentStep < 2 && (
          <button
            className="btn btn-primary"
            onClick={() => setCurrentStep(currentStep + 1)}
            disabled={!canNext()}
            style={{
              padding: '0.6rem 1.5rem',
              opacity: canNext() ? 1 : 0.4,
              cursor: canNext() ? 'pointer' : 'not-allowed',
            }}
          >
            {t('training.btn_next')}
          </button>
        )}

        {currentStep === 2 && (
          <button
            className="btn btn-primary"
            onClick={currentStep === 2 && !loading ? handleTrain : undefined}
            disabled={loading}
            style={{
              padding: '0.6rem 2rem',
              opacity: loading ? 0.6 : 1,
              cursor: loading ? 'not-allowed' : 'pointer',
              border: 'none',
              borderRadius: 8,
              fontWeight: 600,
            }}
          >
            {loading ? t('training.btn_training') : t('training.btn_start')}
          </button>
        )}

        {currentStep === 3 && (
          <button
            className="btn btn-primary"
            onClick={() => setCurrentStep(4)}
            disabled={!result}
            style={{
              padding: '0.6rem 1.5rem',
              opacity: result ? 1 : 0.4,
              cursor: result ? 'pointer' : 'not-allowed',
            }}
          >
            {t('training.btn_summary')}
          </button>
        )}

        {currentStep === 4 && (
          <button
            className="btn"
            onClick={() => {
              setFile(null)
              setCsvPreview(null)
              setResult(null)
              setError(null)
              setCurrentStep(0)
              localStorage.removeItem(TRAINING_STORAGE_KEY)
            }}
            style={{
              padding: '0.6rem 1.5rem',
              background: '#f0f0f0', border: '1px solid #ddd', borderRadius: 8,
              cursor: 'pointer',
            }}
          >
            {t('training.btn_new')}
          </button>
        )}
      </div>
    </>
  )
}
