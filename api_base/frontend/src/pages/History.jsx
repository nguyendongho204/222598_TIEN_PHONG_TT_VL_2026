import { useState, useEffect, useCallback, useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useTranslation } from 'react-i18next'
import { getAllHistories, deleteHistory } from '../services/api'

const checkboxStyle = {  // style cho checkbox
  marginRight: '0.75rem',
  cursor: 'pointer',
  width: 16,
  height: 16,
  accentColor: 'var(--primary)',
}

export default function History() {
  const { t } = useTranslation()
  const [histories, setHistories] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(null)
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [searchText, setSearchText] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [sortField, setSortField] = useState('date')
  const [sortOrder, setSortOrder] = useState('desc')

  const fetchHistories = async () => {  // tải lại lịch sử từ API
    setLoading(true)
    setError(null)
    try {
      const res = await getAllHistories()
      setHistories(res.histories || [])
      setSelectedIds(new Set())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {  // tải lịch sử khi mount
    fetchHistories()
  }, [])

  const handleDelete = async (id) => {  // xoá một bản ghi
    if (!confirm(t('history.confirm_delete_one'))) return
    try {
      await deleteHistory(id)
      setHistories((prev) => prev.filter((h) => h.execution_id !== id))
      setSelectedIds((prev) => { const next = new Set(prev); next.delete(id); return next })
    } catch (err) {
      alert(t('history.err_delete') + ' ' + err.message)
    }
  }

  const handleDeleteSelected = useCallback(async () => {  // xoá nhiều bản ghi
    if (selectedIds.size === 0) return
    if (!confirm(t('history.confirm_delete_multi', { count: selectedIds.size }))) return
    for (const id of selectedIds) {
      try {
        await deleteHistory(id)
      } catch (err) {
        alert(t('history.err_delete_id', { id }) + ' ' + err.message)
      }
    }
    setHistories((prev) => prev.filter((h) => !selectedIds.has(h.execution_id)))
    setSelectedIds(new Set())
  }, [selectedIds])

  const toggleSelect = (id) => {  // chọn/bỏ chọn một bản ghi
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const getAccuracyValue = (history) => {
    const r = history.results || {}
    return r.accuracy ?? 0
  }

  const getDatasetName = (history) => {
    const r = history.results || {}
    return r.dataset_name || (history.file_id || '').replace('.csv', '') || history.execution_id
  }

  const formatTimestamp = (ts) => {
    try {
      return new Date(ts).toLocaleString('vi-VN', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit',
      })
    } catch { return ts }
  }

  const filteredHistories = useMemo(() => {
    let result = [...histories]
    // Search by dataset name
    if (searchText.trim()) {
      const q = searchText.trim().toLowerCase()
      result = result.filter(h => {
        const name = getDatasetName(h).toLowerCase()
        const fileId = (h.file_id || '').toLowerCase()
        return name.includes(q) || fileId.includes(q)
      })
    }
    // Filter by status
    if (statusFilter !== 'all') {
      result = result.filter(h => h.status === statusFilter)
    }
    // Sort
    result.sort((a, b) => {
      let cmp = 0
      if (sortField === 'date') {
        cmp = new Date(a.timestamp) - new Date(b.timestamp)
      } else if (sortField === 'accuracy') {
        cmp = getAccuracyValue(a) - getAccuracyValue(b)
      } else if (sortField === 'name') {
        cmp = getDatasetName(a).localeCompare(getDatasetName(b))
      }
      return sortOrder === 'desc' ? -cmp : cmp
    })
    return result
  }, [histories, searchText, statusFilter, sortField, sortOrder])

  const chartData = useMemo(() =>
    filteredHistories.slice(0, 10).reverse().map((h, i) => ({
      name: `#${i + 1}`,
      accuracy: +(getAccuracyValue(h) * 100).toFixed(2),
    })),
    [filteredHistories]
  )

  const allSelected = selectedIds.size === filteredHistories.length
  const someSelected = selectedIds.size > 0 && selectedIds.size < filteredHistories.length

  const toggleSelectAll = () => {
    if (selectedIds.size === filteredHistories.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filteredHistories.map((h) => h.execution_id)))
    }
  }

  if (loading) {  // hiển thị spinner
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <div className="spinner" style={{ width: 40, height: 40 }} />
        <p style={{ marginTop: '1rem', color: '#888' }}>{t('history.loading')}</p>
      </div>
    )
  }

  if (error) {  // hiển thị lỗi
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <div className="status-bar status-error">{error}</div>
        <button className="btn btn-primary" onClick={fetchHistories}>{t('history.retry')}</button>
      </div>
    )
  }

  if (histories.length === 0) {  // chưa có lịch sử
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <h2>{t('history.empty_title')}</h2>
        <p style={{ color: '#888', marginTop: '1rem' }}>{t('history.empty_msg')}</p>
        <p style={{ color: '#aaa', fontSize: '0.9rem' }}>{t('history.empty_hint')}</p>
      </div>
    )
  }

  return (
    <>
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 style={{ border: 'none', margin: 0, padding: 0 }}>{t('history.title')}</h2>
          <button className="btn btn-primary" onClick={fetchHistories}>{t('history.refresh')}</button>
        </div>
        <p style={{ color: '#888', marginBottom: '1rem' }}>{t('history.total')} {histories.length} {t('history.records')}</p>

        {chartData.length > 1 && (  // biểu đồ xu hướng độ chính xác
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Legend />
              <Bar dataKey="accuracy" name={t('history.accuracy')} fill="#667eea" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Filter bar */}
      <div className="card" style={{ padding: '0.75rem 1rem' }}>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {/* Search */}
          <div style={{ flex: '1 1 200px', position: 'relative' }}>
            <span style={{ position: 'absolute', left: 10, top: 8, fontSize: '0.85rem', color: 'var(--text-muted)' }}>S</span>
            <input
              type="text"
              placeholder={t('history.search_placeholder') || 'Search dataset...'}
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
              style={{
                width: '100%', padding: '0.5rem 0.5rem 0.5rem 2rem',
                border: '1px solid #ddd', borderRadius: 8, fontSize: '0.85rem',
                boxSizing: 'border-box',
              }}
            />
          </div>
          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={e => setStatusFilter(e.target.value)}
            style={{
              padding: '0.5rem 0.75rem', border: '1px solid #ddd', borderRadius: 8,
              fontSize: '0.85rem', background: 'white', cursor: 'pointer',
            }}
          >
            <option value="all">{t('history.filter_all') || 'All status'}</option>
            <option value="success">{t('history.filter_success') || 'Success'}</option>
            <option value="error">{t('history.filter_error') || 'Error'}</option>
          </select>
          {/* Sort field */}
          <select
            value={sortField}
            onChange={e => setSortField(e.target.value)}
            style={{
              padding: '0.5rem 0.75rem', border: '1px solid #ddd', borderRadius: 8,
              fontSize: '0.85rem', background: 'white', cursor: 'pointer',
            }}
          >
            <option value="date">{t('history.sort_date') || 'Date'}</option>
            <option value="accuracy">{t('history.sort_accuracy') || 'Accuracy'}</option>
            <option value="name">{t('history.sort_name') || 'Name'}</option>
          </select>
          {/* Sort order */}
          <button
            onClick={() => setSortOrder(prev => prev === 'desc' ? 'asc' : 'desc')}
            style={{
              padding: '0.5rem 0.75rem', border: '1px solid #ddd', borderRadius: 8,
              fontSize: '0.85rem', background: 'white', cursor: 'pointer',
              minWidth: 40, display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
            title={sortOrder === 'desc' ? 'Newest first' : 'Oldest first'}
          >
            {sortOrder === 'desc' ? '↓' : '↑'}
          </button>
          {/* Result count */}
          <span style={{ fontSize: '0.8rem', color: '#888', marginLeft: 'auto' }}>
            {filteredHistories.length}/{histories.length} {t('history.records')}
          </span>
        </div>
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              checked={allSelected}
              ref={(el) => { if (el) el.indeterminate = someSelected }}
              onChange={toggleSelectAll}
              style={checkboxStyle}
            />
            <span style={{ fontSize: '0.85rem', color: '#666', userSelect: 'none' }}>
              {selectedIds.size > 0 ? `${t('history.selected')} ${selectedIds.size}` : t('history.select_all')}
            </span>
          </div>
          <button
            className="btn btn-danger"
            disabled={selectedIds.size === 0}
            onClick={handleDeleteSelected}
            style={{ opacity: selectedIds.size === 0 ? 0.5 : 1, padding: '0.3rem 0.8rem', fontSize: '0.85rem' }}
          >
            {t('history.delete_selected')}{selectedIds.size > 0 ? ` (${selectedIds.size})` : ''}
          </button>
        </div>

        {filteredHistories.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: '#999' }}>
            {t('history.no_results') || 'No records match your filter'}
          </div>
        ) : filteredHistories.map((h) => {
          const acc = getAccuracyValue(h)
          const name = getDatasetName(h)
          const isExpanded = expanded === h.execution_id
          const isSelected = selectedIds.has(h.execution_id)
          return (
            <div
              key={h.execution_id}
              style={{
                padding: '0.75rem 1rem',
                marginBottom: '0.5rem',
                border: `1px solid ${isSelected ? 'var(--primary)' : isExpanded ? 'var(--primary-light)' : 'var(--border)'}`,
                borderRadius: 8,
                cursor: 'pointer',
                background: isSelected ? 'var(--primary-light)' : isExpanded ? 'var(--bg-card-alt)' : 'var(--bg-card)',
                transition: 'all 0.15s',
              }}
              onClick={() => setExpanded(isExpanded ? null : h.execution_id)}
            >
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleSelect(h.execution_id)}
                  onClick={(e) => e.stopPropagation()}
                  style={checkboxStyle}
                />
                <div style={{ flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <strong style={{ fontSize: '1rem' }}>{name}</strong>
                    <div style={{ color: '#888', fontSize: '0.8rem', marginTop: '0.2rem' }}>
                      {h.algorithm_name || 'Training'} — {formatTimestamp(h.timestamp)}
                      {h.results?.samples ? ` — ${h.results.samples} mẫu` : ''}
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
                    <span className={`badge ${acc >= 0.85 ? 'badge-success' : acc >= 0.7 ? 'badge-info' : 'badge-danger'}`}
                      style={{ fontSize: '0.9rem', padding: '0.3rem 0.6rem' }}>
                      {(acc * 100).toFixed(1)}%
                    </span>
                    <span className={`badge ${h.status === 'success' ? 'badge-success' : 'badge-danger'}`}
                      style={{ fontSize: '0.75rem' }}>
                      {h.status}
                    </span>
                    <button
                      className="btn btn-danger"
                      style={{ padding: '0.3rem 0.6rem', fontSize: '0.8rem' }}
                      onClick={(e) => { e.stopPropagation(); handleDelete(h.execution_id) }}
                    >
                      {t('history.delete')}
                    </button>
                  </div>
                </div>
              </div>

              {isExpanded && (() => {
                const r = h.results || {}
                const acc = r.accuracy ?? 0
                const accStd = r.accuracy_std ?? 0
                const datasetName = r.dataset_name || (h.file_id || '').replace('.csv', '') || 'Dataset'

                const barData = [
                  { name: 'JEPA+SVM', accuracy: +(acc * 100).toFixed(2) },
                ]

                return (
                  <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #e8e8ff' }}>
                    {/* Parameters summary */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', fontSize: '0.8rem', color: '#888' }}>
                        {h.parameters && Object.entries(h.parameters).map(([k, v]) => (
                          <span key={k} style={{ background: '#f0f0ff', padding: '0.2rem 0.5rem', borderRadius: 4 }}>
                            {k}: {typeof v === 'number' ? (k.includes('size') || k.includes('variance') || k.includes('weight') ? `${(v * 100).toFixed(0)}%` : v) : v}
                          </span>
                        ))}
                        {h.execution_time != null && (
                          <span style={{ background: '#f0f0ff', padding: '0.2rem 0.5rem', borderRadius: 4 }}>
                            {h.execution_time.toFixed(2)}s
                          </span>
                        )}
                      </div>
                      <span className={`badge ${h.status === 'success' ? 'badge-success' : 'badge-danger'}`}>{h.status}</span>
                    </div>

                    {/* Status */}
                    <div className="status-bar status-success" style={{ marginBottom: '0.75rem' }}>
                      {t('training.complete')}
                    </div>

                    {/* Metric grid */}
                    <div className="metric-grid" style={{ marginBottom: '1rem' }}>
                      <div className="metric-card">
                        <div className="metric-value" style={{ fontSize: '1rem' }}>{datasetName}</div>
                        <div className="metric-label">{t('training.dataset')}</div>
                      </div>
                      <div className="metric-card">
                        <div className="metric-value">{r.samples ?? '—'}</div>
                        <div className="metric-label">{t('training.samples')}</div>
                      </div>
                      <div className="metric-card">
                        <div className="metric-value">{r.features ?? '—'}</div>
                        <div className="metric-label">{t('training.features')}</div>
                      </div>
                      <div className="metric-card">
                        <div className="metric-value">{r.classes ?? '—'}</div>
                        <div className="metric-label">{t('training.classes')}</div>
                      </div>
                      {r.pca_components != null && (
                        <div className="metric-card">
                          <div className="metric-value">{r.pca_components}</div>
                          <div className="metric-label">{t('training.pca_components')}</div>
                        </div>
                      )}
                    </div>

                    {/* JEPA+SVM accuracy */}
                    <h2 style={{ fontSize: '1rem', marginBottom: '0.75rem' }}>{t('training.comparison')}</h2>
                    <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                      <div className="metric-card" style={{ display: 'inline-block', border: '2px solid #28a745' }}>
                        <div className="metric-label" style={{ color: '#28a745' }}>JEPA+SVM</div>
                        <div className="metric-value" style={{ fontSize: '1rem' }}>
                          {(acc * 100).toFixed(2)}% ± {(accStd * 100).toFixed(2)}%
                        </div>
                      </div>
                    </div>

                    {/* Bar chart */}
                    {barData.length > 0 && (
                      <ResponsiveContainer width="100%" height={280} style={{ marginBottom: '1rem' }}>
                        <BarChart data={barData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                          <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                          <Tooltip formatter={(v) => [`${v}%`, 'Accuracy']} />
                          <Bar dataKey="accuracy" fill="#28a745" radius={[6, 6, 0, 0]} barSize={50} />
                        </BarChart>
                      </ResponsiveContainer>
                    )}

                    {/* Classification report */}
                    {r.classification_report && (
                      <div style={{ marginBottom: '1rem' }}>
                        <h2 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>{t('training.class_report')}</h2>
                        <div style={{ overflowX: 'auto' }}>
                          <table className="metrics-table" style={{ fontSize: '0.85rem' }}>
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
                              {Object.entries(r.classification_report).map(([key, val]) => {
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

                    {/* Confusion matrix */}
                    {r.confusion_matrix && r.confusion_matrix.length > 0 && (
                      <div>
                        <h2 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>{t('training.conf_matrix')}</h2>
                        <p style={{ color: '#888', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
                          {t('training.conf_desc')}
                        </p>
                        <div style={{ overflowX: 'auto' }}>
                          <table className="confusion-matrix" style={{ fontSize: '0.85rem' }}>
                            <thead>
                              <tr>
                                <th></th>
                                {r.confusion_matrix[0].map((_, i) => (
                                  <th key={i}>{t('training.pred')} {i}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {r.confusion_matrix.map((row, i) => (
                                <tr key={i}>
                                  <td><strong>{t('training.true')} {i}</strong></td>
                                  {row.map((val, j) => {
                                    const maxVal = Math.max(...r.confusion_matrix.map(rr => Math.max(...rr)))
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
                                        padding: '0.5rem 0.8rem',
                                        minWidth: 50,
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
                  </div>
                )
              })()}
            </div>
          )
        })}
      </div>
    </>
  )
}
