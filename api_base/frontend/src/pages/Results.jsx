import { useState, useEffect } from 'react'  // hook trạng thái và lifecycle
import { useTranslation } from 'react-i18next'  // đa ngôn ngữ
import { getAllHistories } from '../services/api'  // API lấy lịch sử
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'  // biểu đồ cột

export default function Results() {
  const { t } = useTranslation()  // hàm dịch
  const [histories, setHistories] = useState([])  // danh sách lịch sử
  const [loading, setLoading] = useState(true)  // trạng thái đang tải

  useEffect(() => {  // gọi API khi mount
    getAllHistories()
      .then(res => setHistories(res.histories || []))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) {  // hiển thị spinner
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <div className="spinner" style={{ width: 40, height: 40 }} />
        <p style={{ marginTop: '1rem', color: '#888' }}>{t('results.loading')}</p>
      </div>
    )
  }

  const latest = {}  // nhóm theo file_id, giữ bản ghi mới nhất
  histories.forEach(h => {
    const name = (h.file_id || '').replace('.csv', '')
    if (!latest[name] || new Date(h.timestamp) > new Date(latest[name].timestamp)) {
      latest[name] = h
    }
  })

  const items = Object.entries(latest)  // chuyển thành mảng để hiển thị
    .filter(([, h]) => h.results)
    .map(([name, h]) => {
      const r = h.results
      const acc = r.accuracy ?? r.final_accuracy ?? r.ensemble_accuracy ?? r.svm_accuracy ?? 0
      return { name, acc, version: r._version || '-' }
    })
    .sort((a, b) => a.name.localeCompare(b.name))

  if (items.length === 0) {  // chưa có dữ liệu
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <h2>{t('results.title')}</h2>
        <p style={{ color: '#888', marginTop: '1rem' }}>{t('results.empty')}</p>
      </div>
    )
  }

  const chartData = items.map(d => ({  // dữ liệu cho biểu đồ
    name: d.name,
    'JEPA+SVM': +(d.acc * 100).toFixed(2),
  }))

  const avgAcc = items.reduce((s, d) => s + d.acc, 0) / items.length

  return (
    <>
      <div className="card">
        <h2>{t('results.title')}</h2>
        <p style={{ color: '#888', marginBottom: '1rem', lineHeight: 1.6 }}>
          {t('results.desc', { count: items.length, avg: (avgAcc * 100).toFixed(2) })}
        </p>

        <div style={{ overflowX: 'auto' }}>  {/* bảng kết quả */}
          <table className="metrics-table">
            <thead>
              <tr>
                <th>{t('results.col_dataset')}</th>
                <th>{t('results.col_ensemble')}</th>
                <th>{t('results.col_version')}</th>
              </tr>
            </thead>
            <tbody>
              {items.map((d) => (
                <tr key={d.name}>
                  <td><strong>{d.name}</strong></td>
                  <td style={{ fontWeight: 700, color: 'var(--success)' }}>
                    {(d.acc * 100).toFixed(2)}%
                  </td>
                  <td style={{ fontSize: '0.85rem' }}>{d.version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="conclusion-box">
          <strong>{t('results.conclusion', { count: items.length, avg: (avgAcc * 100).toFixed(2) })}</strong>
        </div>
      </div>

      {items.length > 1 && (  // biểu đồ so sánh khi có nhiều dataset
        <div className="card">
          <h2>{t('results.chart_title')}</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-20} textAnchor="end" tick={{ fontSize: 12 }} />
              <YAxis domain={[50, 100]} tickFormatter={(v) => `${v}%`} />
              <Tooltip formatter={(v) => `${v.toFixed(2)}%`} />
              <Bar dataKey="JEPA+SVM" fill="#28a745" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )
}
