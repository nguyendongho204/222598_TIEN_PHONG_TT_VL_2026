import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { healthCheck, getApiInfo, getEnsembleInfo } from '../services/api'

const LITERATURE = {
  'Balance':        { paper: '91.48%', method: 'SVC (IJCA WEKA 2012)', ref: 'WEKA benchmark' },
  'Banknote':       { paper: '100.0%', method: 'SVM RBF', ref: 'Multiple sources (2018-2026)' },
  'Breast Cancer':  { paper: '97.07%', method: 'SVM (PLOS One 2025)', ref: '10.1371/journal.pone.0326221' },
  'Car':            { paper: '99.00%', method: 'SVM tuned (chiayenho 2022)', ref: 'GitHub (unverified)' },
  'Dermatology':    { paper: '98.61%', method: 'SVM optimized (Dhanyaa24 2025)', ref: 'GitHub (unverified)' },
  'Ecoli':          { paper: '87.16%', method: 'SVM (Das 2024)', ref: 'CEUR-WS Vol-3664 ✅' },
  'Glass':          { paper: '99.25%', method: 'SVM+BayesOpt (Zhou 2023)', ref: 'HSET 39 ✅' },
  'Haberman':       { paper: '74.44%', method: 'SVM (Aljawad 2017)', ref: 'JTAIT' },
  'Heart':          { paper: '99.75%', method: 'Boosting SVM (Owusu 2021)', ref: 'PMC8718315 ✅' },
  'Iris':           { paper: '100.0%', method: 'SVM RBF (Aeberhard 1992)', ref: '10.1007/BF00116832' },
  'Liver':          { paper: '72.00%', method: 'SVM (Forsyth 1990)', ref: 'UCI BUPA' },
  'Mushroom':       { paper: '100.0%', method: 'SVM', ref: 'Multiple sources' },
  'Optical':        { paper: '97.00%', method: 'SVM RBF (sklearn 2011)', ref: 'scikit-learn example' },
  'Sonar':          { paper: '90.48%', method: 'SVM+feat reduction (Wenkel 2018)', ref: 'UCI Sonar' },
  'Spambase':       { paper: '93.13%', method: 'C-SVC (Takci 2023)', ref: 'IAJIT 20(1) ✅' },
  'Wine':           { paper: '100.0%', method: 'SVM RBF (Aeberhard 1992)', ref: '10.1007/BF00116832' },
  'Wine Quality':   { paper: '92.50%', method: 'SVM tuned (Cortez 2009)', ref: 'DSS 47(4)' },
  'Yeast':          { paper: '98.32%', method: 'SVM tuned (Martinaa1408 2025)', ref: 'GitHub (unverified)' },
}

const NO_SVM_NOTE = 'Không có SVM reference (published best: Neural Network / Decision Tree / Bayes)'

function verdict(jepaAcc, paperAcc) {
  const diff = jepaAcc - paperAcc
  if (diff > 1.0) return { text: 'Win', cls: 'tag-win', diff: `+${diff.toFixed(1)}%` }
  if (diff < -1.0) return { text: 'Loss', cls: 'tag-loss', diff: `${diff.toFixed(1)}%` }
  return { text: 'Tie', cls: 'tag-tie', diff: `±0.0%` }
}

export default function Dashboard() {
  const { t } = useTranslation()  // hàm dịch
  const navigate = useNavigate()  // điều hướng
  const [health, setHealth] = useState(null)  // trạng thái health API
  const [apiInfo, setApiInfo] = useState(null)  // thông tin cấu hình API
  const [ensembleInfo, setEnsembleInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [selectedUci, setSelectedUci] = useState(null)

  useEffect(() => {  // gọi API khi component mount
    async function load() {
      try {
        const [h, info, ens] = await Promise.all([  // gọi đồng thời 3 API
          healthCheck(),
          getApiInfo(),
          getEnsembleInfo(),
        ])
        setHealth(h)  // lưu kết quả health
        setApiInfo(info)  // lưu thông tin API
        setEnsembleInfo(ens)  // lưu thông tin ensemble
      } catch (err) {
        setError(err.message)  // lưu lỗi
      } finally {
        setLoading(false)  // kết thúc tải
      }
    }
    load()  // chạy hàm load
  }, [])  // [] = chỉ chạy một lần

  if (loading) {  // hiển thị spinner khi đang tải
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <div className="spinner" style={{ width: 40, height: 40 }} />
        <p style={{ marginTop: '1rem', color: '#888' }}>{t('dashboard.loading')}</p>
      </div>
    )
  }

  if (error) {  // hiển thị lỗi nếu có
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <div className="status-bar status-error">{t('dashboard.failed')} {error}</div>
        <button className="btn btn-primary" onClick={() => window.location.reload()}>
          {t('dashboard.retry')}
        </button>
      </div>
    )
  }

  return (
    <>
      <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
        <p style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--primary)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.75rem' }}>
          {t('dashboard.badge')}
        </p>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text)', marginBottom: '0.5rem' }}>
          {t('dashboard.title')}
        </h1>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', maxWidth: 500, margin: '0 auto 1.25rem', lineHeight: 1.6 }}>
          {t('dashboard.subtitle')}
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <button className="btn btn-primary" onClick={() => navigate('/training')}>
            {t('dashboard.start_training')}
          </button>
          <button className="btn" onClick={() => navigate('/history')}>
            {t('dashboard.view_history')}
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
          {[
            { num: 1, label: t('dashboard.flow_upload') },
            { num: 2, label: t('dashboard.flow_config') },
            { num: 3, label: t('dashboard.flow_train') },
            { num: 4, label: t('dashboard.flow_results') },
          ].map((step, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              {i > 0 && <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>/</span>}
              <span style={{
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                width: 22, height: 22, borderRadius: '50%',
                background: 'var(--primary-light)', color: 'var(--primary)',
                fontSize: '0.7rem', fontWeight: 700,
              }}>{step.num}</span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{step.label}</span>
            </div>
          ))}
        </div>
      </div>

        <div className="card" style={{ overflowX: 'auto' }}>
        <h2>{t('dashboard.table_title')}</h2>
        <p style={{ color: '#888', fontSize: '0.85rem', marginBottom: '0.8rem' }}>Stratified K-Fold (5 folds) · JEPA self-supervised + supervised fine-tune · Click dataset để xem so sánh với published SVM</p>
        <table className="metrics-table" style={{ minWidth: 500 }}>
          <thead>
            <tr>
              <th>{t('dashboard.col_dataset')}</th>
              <th>JEPA+SVM</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
            </tr>
          </thead>
          <tbody>
              {[
                { name: 'Iris', acc: 95.33, p: 90.24, r: 90.00, f: 89.97 },
                { name: 'Wine', acc: 98.89, p: 100.00, r: 100.00, f: 100.00 },
                { name: 'Breast Cancer', acc: 97.19, p: 98.63, r: 97.62, f: 98.09 },
                { name: 'Wine Quality', acc: 75.67, p: 69.29, r: 69.39, f: 69.24 },
              ].map((row) => {
              const isSelected = selectedDataset === row.name
              return (
                <tr
                  key={row.name}
                  onClick={() => setSelectedDataset(isSelected ? null : row.name)}
                  style={{
                    cursor: 'pointer',
                    background: isSelected ? 'var(--primary-light)' : '',
                    borderBottom: isSelected ? '2px solid var(--primary)' : '',
                  }}
                >
                  <td><strong>{row.name}</strong></td>
                  <td style={{ fontWeight: 700 }}>{row.acc.toFixed(2)}%</td>
                  <td>{row.p.toFixed(2)}%</td>
                  <td>{row.r.toFixed(2)}%</td>
                  <td>{row.f.toFixed(2)}%</td>
                </tr>
              )
            })}
          </tbody>
        </table>

        <div className="card" style={{ marginTop: '1.5rem' }}>
          <h3 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>JEPA+SVM vs Published SVM — Benchmark (23 UCI Datasets)</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            JEPA+SVM: Stratified K-Fold (5 folds), MinMaxScaler(-1,1) · JEPA self-supervised + supervised fine-tune
          </p>
          <style>{`
            .tag-win { background: #e8f5e9; color: #2e7d32; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
            .tag-loss { background: #fce4ec; color: #c62828; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
            .tag-tie { background: #fff8e1; color: #f57f17; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
          `}</style>
          <table className="metrics-table" style={{ minWidth: 800 }}>
            <thead>
              <tr>
                <th>Dataset</th><th>n</th><th>Feat</th><th>Cls</th>
                <th>JEPA+SVM</th><th>Paper SVM</th><th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {[
                {d:'Abalone',n:4177,f:8,c:28,a:27.12,s:2.16,p:12.04,r:12.71,f1:12.14,t:1469},
                {d:'Balance',n:625,f:4,c:3,a:96.48,s:1.30,p:86.67,r:96.55,f1:89.89,t:303},
                {d:'Banknote',n:1372,f:4,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:321},
                {d:'Breast Cancer',n:569,f:30,c:2,a:97.19,s:1.70,p:98.63,r:97.62,f1:98.09,t:235},
                {d:'Car',n:1728,f:6,c:4,a:98.32,s:0.72,p:98.82,r:97.35,f1:98.05,t:453},
                {d:'Dermatology',n:366,f:34,c:6,a:97.81,s:0.68,p:97.22,r:97.22,f1:96.97,t:167},
                {d:'Ecoli',n:336,f:8,c:8,a:95.84,s:2.16,p:97.13,r:94.29,f1:95.17,t:193},
                {d:'Glass',n:214,f:10,c:6,a:93.93,s:1.11,p:94.79,r:95.48,f1:94.61,t:100},
                {d:'Haberman',n:306,f:3,c:2,a:69.91,s:4.10,p:58.24,r:55.83,f1:56.01,t:133},
                {d:'Heart',n:303,f:13,c:5,a:54.79,s:4.52,p:20.91,r:24.26,f1:22.30,t:136},
                {d:'Ionosphere',n:351,f:34,c:2,a:94.01,s:1.91,p:93.56,r:90.89,f1:91.99,t:152},
                {d:'Iris',n:150,f:4,c:3,a:95.33,s:4.52,p:90.24,r:90.00,f1:89.97,t:68},
                {d:'Liver',n:345,f:6,c:2,a:72.17,s:7.69,p:62.44,r:62.76,f1:62.12,t:151},
                {d:'Mushroom',n:8124,f:22,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:1083},
                {d:'Optical',n:5620,f:64,c:10,a:98.81,s:0.20,p:98.69,r:98.66,f1:98.67,t:1039},
                {d:'Page Blocks',n:5473,f:10,c:5,a:96.80,s:0.68,p:88.53,r:86.90,f1:86.91,t:588},
                {d:'Sonar',n:208,f:60,c:2,a:84.13,s:1.16,p:85.29,r:85.29,f1:85.29,t:66},
                {d:'Spambase',n:4601,f:57,c:2,a:93.31,s:0.39,p:92.49,r:91.97,f1:92.21,t:975},
                {d:'Vehicle',n:846,f:18,c:5,a:79.90,s:2.71,p:64.54,r:65.35,f1:63.84,t:203},
                {d:'Waveform',n:5000,f:21,c:3,a:87.06,s:0.57,p:87.63,r:87.49,f1:87.43,t:556},
                {d:'Wine',n:178,f:13,c:3,a:98.89,s:1.36,p:100.0,r:100.0,f1:100.0,t:97},
                {d:'Wine Quality',n:1599,f:11,c:2,a:75.67,s:3.74,p:69.29,r:69.39,f1:69.24,t:461},
                {d:'Yeast',n:1484,f:9,c:10,a:61.73,s:2.39,p:56.29,r:60.25,f1:56.92,t:347},
              ].map((row, idx) => {
                const isSelected = selectedUci === row.d
                const lit = LITERATURE[row.d]
                const paperAcc = lit ? parseFloat(lit.paper) : null
                const v = paperAcc !== null ? verdict(row.a, paperAcc) : null
                return (
                  <tr key={idx}
                    onClick={() => setSelectedUci(isSelected ? null : row.d)}
                    style={{ cursor: 'pointer', background: isSelected ? 'var(--primary-light)' : '' }}
                  >
                    <td><strong>{row.d}</strong></td>
                    <td>{row.n}</td><td>{row.f}</td><td>{row.c}</td>
                    <td style={{ fontWeight: 700 }}>{row.a.toFixed(2)}%</td>
                    <td style={{ color: lit ? 'var(--text)' : '#999' }}>{lit ? lit.paper : 'N/A'}</td>
                    <td>{v ? <span className={v.cls}>{v.diff} {v.text}</span> : <span style={{color:'#999',fontSize:'0.75rem'}}>—</span>}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {(() => {
            const withLit = Object.keys(LITERATURE)
            const rows = [{d:'Abalone',n:4177,f:8,c:28,a:27.12,s:2.16,p:12.04,r:12.71,f1:12.14,t:1469},
{d:'Balance',n:625,f:4,c:3,a:96.48,s:1.30,p:86.67,r:96.55,f1:89.89,t:303},
{d:'Banknote',n:1372,f:4,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:321},
{d:'Breast Cancer',n:569,f:30,c:2,a:97.19,s:1.70,p:98.63,r:97.62,f1:98.09,t:235},
{d:'Car',n:1728,f:6,c:4,a:98.32,s:0.72,p:98.82,r:97.35,f1:98.05,t:453},
{d:'Dermatology',n:366,f:34,c:6,a:97.81,s:0.68,p:97.22,r:97.22,f1:96.97,t:167},
{d:'Ecoli',n:336,f:8,c:8,a:95.84,s:2.16,p:97.13,r:94.29,f1:95.17,t:193},
{d:'Glass',n:214,f:10,c:6,a:93.93,s:1.11,p:94.79,r:95.48,f1:94.61,t:100},
{d:'Haberman',n:306,f:3,c:2,a:69.91,s:4.10,p:58.24,r:55.83,f1:56.01,t:133},
{d:'Heart',n:303,f:13,c:5,a:54.79,s:4.52,p:20.91,r:24.26,f1:22.30,t:136},
{d:'Ionosphere',n:351,f:34,c:2,a:94.01,s:1.91,p:93.56,r:90.89,f1:91.99,t:152},
{d:'Iris',n:150,f:4,c:3,a:95.33,s:4.52,p:90.24,r:90.00,f1:89.97,t:68},
{d:'Liver',n:345,f:6,c:2,a:72.17,s:7.69,p:62.44,r:62.76,f1:62.12,t:151},
{d:'Mushroom',n:8124,f:22,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:1083},
{d:'Optical',n:5620,f:64,c:10,a:98.81,s:0.20,p:98.69,r:98.66,f1:98.67,t:1039},
{d:'Page Blocks',n:5473,f:10,c:5,a:96.80,s:0.68,p:88.53,r:86.90,f1:86.91,t:588},
{d:'Sonar',n:208,f:60,c:2,a:84.13,s:1.16,p:85.29,r:85.29,f1:85.29,t:66},
{d:'Spambase',n:4601,f:57,c:2,a:93.31,s:0.39,p:92.49,r:91.97,f1:92.21,t:975},
{d:'Vehicle',n:846,f:18,c:5,a:79.90,s:2.71,p:64.54,r:65.35,f1:63.84,t:203},
{d:'Waveform',n:5000,f:21,c:3,a:87.06,s:0.57,p:87.63,r:87.49,f1:87.43,t:556},
{d:'Wine',n:178,f:13,c:3,a:98.89,s:1.36,p:100.0,r:100.0,f1:100.0,t:97},
{d:'Wine Quality',n:1599,f:11,c:2,a:75.67,s:3.74,p:69.29,r:69.39,f1:69.24,t:461},
{d:'Yeast',n:1484,f:9,c:10,a:61.73,s:2.39,p:56.29,r:60.25,f1:56.92,t:347}]
            const comp = rows.filter(r => withLit.includes(r.d))
            const wins = comp.filter(r => { const l = LITERATURE[r.d]; return r.a - parseFloat(l.paper) > 1.0 }).length
            const losses = comp.filter(r => { const l = LITERATURE[r.d]; return parseFloat(l.paper) - r.a > 1.0 }).length
            const ties = comp.length - wins - losses
            const avgJepa = comp.reduce((s, r) => s + r.a, 0) / comp.length
            const avgPaper = comp.reduce((s, r) => s + parseFloat(LITERATURE[r.d].paper), 0) / comp.length
            return (
              <div className="conclusion-box" style={{ marginTop: '0.8rem' }}>
                <strong>JEPA+SVM avg: {avgJepa.toFixed(2)}%</strong> vs <strong>Published SVM avg: {avgPaper.toFixed(2)}%</strong>
                &nbsp;·&nbsp; Wins: <strong style={{color:'#2e7d32'}}>{wins}</strong> / Losses: <strong style={{color:'#c62828'}}>{losses}</strong> / Ties: <strong>{ties}</strong> (of {comp.length} datasets with SVM literature)
              </div>
            )
          })()}

          {selectedUci && (() => {
            const row = [{d:'Abalone',n:4177,f:8,c:28,a:27.12,s:2.16,p:12.04,r:12.71,f1:12.14,t:1469},
{d:'Balance',n:625,f:4,c:3,a:96.48,s:1.30,p:86.67,r:96.55,f1:89.89,t:303},
{d:'Banknote',n:1372,f:4,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:321},
{d:'Breast Cancer',n:569,f:30,c:2,a:97.19,s:1.70,p:98.63,r:97.62,f1:98.09,t:235},
{d:'Car',n:1728,f:6,c:4,a:98.32,s:0.72,p:98.82,r:97.35,f1:98.05,t:453},
{d:'Dermatology',n:366,f:34,c:6,a:97.81,s:0.68,p:97.22,r:97.22,f1:96.97,t:167},
{d:'Ecoli',n:336,f:8,c:8,a:95.84,s:2.16,p:97.13,r:94.29,f1:95.17,t:193},
{d:'Glass',n:214,f:10,c:6,a:93.93,s:1.11,p:94.79,r:95.48,f1:94.61,t:100},
{d:'Haberman',n:306,f:3,c:2,a:69.91,s:4.10,p:58.24,r:55.83,f1:56.01,t:133},
{d:'Heart',n:303,f:13,c:5,a:54.79,s:4.52,p:20.91,r:24.26,f1:22.30,t:136},
{d:'Ionosphere',n:351,f:34,c:2,a:94.01,s:1.91,p:93.56,r:90.89,f1:91.99,t:152},
{d:'Iris',n:150,f:4,c:3,a:95.33,s:4.52,p:90.24,r:90.00,f1:89.97,t:68},
{d:'Liver',n:345,f:6,c:2,a:72.17,s:7.69,p:62.44,r:62.76,f1:62.12,t:151},
{d:'Mushroom',n:8124,f:22,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:1083},
{d:'Optical',n:5620,f:64,c:10,a:98.81,s:0.20,p:98.69,r:98.66,f1:98.67,t:1039},
{d:'Page Blocks',n:5473,f:10,c:5,a:96.80,s:0.68,p:88.53,r:86.90,f1:86.91,t:588},
{d:'Sonar',n:208,f:60,c:2,a:84.13,s:1.16,p:85.29,r:85.29,f1:85.29,t:66},
{d:'Spambase',n:4601,f:57,c:2,a:93.31,s:0.39,p:92.49,r:91.97,f1:92.21,t:975},
{d:'Vehicle',n:846,f:18,c:5,a:79.90,s:2.71,p:64.54,r:65.35,f1:63.84,t:203},
{d:'Waveform',n:5000,f:21,c:3,a:87.06,s:0.57,p:87.63,r:87.49,f1:87.43,t:556},
{d:'Wine',n:178,f:13,c:3,a:98.89,s:1.36,p:100.0,r:100.0,f1:100.0,t:97},
{d:'Wine Quality',n:1599,f:11,c:2,a:75.67,s:3.74,p:69.29,r:69.39,f1:69.24,t:461},
{d:'Yeast',n:1484,f:9,c:10,a:61.73,s:2.39,p:56.29,r:60.25,f1:56.92,t:347}].find(r => r.d === selectedUci)
            if (!row) return null
            const lit = LITERATURE[row.d]
            return (
              <div className="card" style={{ marginTop: '1rem', borderColor: 'var(--primary)' }}>
                <h3 style={{ color: 'var(--primary)' }}>{row.d}</h3>
                <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginTop: '0.5rem' }}>
                  <div className="metric-card">
                    <div className="metric-value">{row.a.toFixed(2)}%</div>
                    <div className="metric-label">JEPA+SVM Accuracy</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value" style={{ fontSize: '0.9rem' }}>±{row.s.toFixed(2)}%</div>
                    <div className="metric-label">Std Dev</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{row.p.toFixed(2)}%</div>
                    <div className="metric-label">Precision (macro)</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{row.r.toFixed(2)}%</div>
                    <div className="metric-label">Recall (macro)</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{row.f1.toFixed(2)}%</div>
                    <div className="metric-label">F1 (macro)</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{row.n}</div>
                    <div className="metric-label">Samples</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{row.f}</div>
                    <div className="metric-label">Features</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{row.c}</div>
                    <div className="metric-label">Classes</div>
                  </div>
                </div>
                {lit && (
                  <div className="info-box" style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                    <strong>Published SVM:</strong> {lit.paper} ({lit.method}) · <em>{lit.ref}</em><br/>
                    <strong>Verdict:</strong> {(() => { const v = verdict(row.a, parseFloat(lit.paper)); return <span className={v.cls}>{v.diff} {v.text}</span> })()}
                  </div>
                )}
                {!lit && (
                  <div className="notice-box" style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                    {NO_SVM_NOTE}
                  </div>
                )}
                <div className="info-box" style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                  Training time: {row.t >= 60 ? `${(row.t/60).toFixed(1)} min` : `${row.t.toFixed(0)}s`} · 5-Fold Cross Validation
                </div>
                <button className="btn" style={{ marginTop: '0.5rem', fontSize: '0.8rem', padding: '0.3rem 0.8rem' }}
                  onClick={() => setSelectedUci(null)}>Đóng</button>
              </div>
            )
          })()}
        </div>

        {selectedDataset && LITERATURE[selectedDataset] && (
          <div className="card" style={{ marginTop: '1.5rem', borderColor: 'var(--primary)' }}>
            <h3 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>
              So sánh: JEPA+SVM vs Published SVM — {selectedDataset}
            </h3>
            <table className="metrics-table" style={{ minWidth: 700 }}>
              <thead>
                <tr>
                  <th>Phương pháp</th>
                  <th>Accuracy</th>
                  <th>Precision (macro)</th>
                  <th>Recall (macro)</th>
                  <th>F1 (macro)</th>
                  <th>Protocol</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { label: `${LITERATURE[selectedDataset].method} (Published)`, acc: LITERATURE[selectedDataset].paper, p: '-', r: '-', f: '-', protocol: 'Paper split', isPaper: true },
                  { label: 'JEPA+SVM (Pipeline của bạn)', acc: `${(() => { const found = [{d:'Abalone',n:4177,f:8,c:28,a:27.12,s:2.16,p:12.04,r:12.71,f1:12.14,t:1469},{d:'Balance',n:625,f:4,c:3,a:96.48,s:1.30,p:86.67,r:96.55,f1:89.89,t:303},{d:'Banknote',n:1372,f:4,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:321},{d:'Breast Cancer',n:569,f:30,c:2,a:97.19,s:1.70,p:98.63,r:97.62,f1:98.09,t:235},{d:'Car',n:1728,f:6,c:4,a:98.32,s:0.72,p:98.82,r:97.35,f1:98.05,t:453},{d:'Dermatology',n:366,f:34,c:6,a:97.81,s:0.68,p:97.22,r:97.22,f1:96.97,t:167},{d:'Ecoli',n:336,f:8,c:8,a:95.84,s:2.16,p:97.13,r:94.29,f1:95.17,t:193},{d:'Glass',n:214,f:10,c:6,a:93.93,s:1.11,p:94.79,r:95.48,f1:94.61,t:100},{d:'Haberman',n:306,f:3,c:2,a:69.91,s:4.10,p:58.24,r:55.83,f1:56.01,t:133},{d:'Heart',n:303,f:13,c:5,a:54.79,s:4.52,p:20.91,r:24.26,f1:22.30,t:136},{d:'Ionosphere',n:351,f:34,c:2,a:94.01,s:1.91,p:93.56,r:90.89,f1:91.99,t:152},{d:'Iris',n:150,f:4,c:3,a:95.33,s:4.52,p:90.24,r:90.00,f1:89.97,t:68},{d:'Liver',n:345,f:6,c:2,a:72.17,s:7.69,p:62.44,r:62.76,f1:62.12,t:151},{d:'Mushroom',n:8124,f:22,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:1083},{d:'Optical',n:5620,f:64,c:10,a:98.81,s:0.20,p:98.69,r:98.66,f1:98.67,t:1039},{d:'Page Blocks',n:5473,f:10,c:5,a:96.80,s:0.68,p:88.53,r:86.90,f1:86.91,t:588},{d:'Sonar',n:208,f:60,c:2,a:84.13,s:1.16,p:85.29,r:85.29,f1:85.29,t:66},{d:'Spambase',n:4601,f:57,c:2,a:93.31,s:0.39,p:92.49,r:91.97,f1:92.21,t:975},{d:'Vehicle',n:846,f:18,c:5,a:79.90,s:2.71,p:64.54,r:65.35,f1:63.84,t:203},{d:'Waveform',n:5000,f:21,c:3,a:87.06,s:0.57,p:87.63,r:87.49,f1:87.43,t:556},{d:'Wine',n:178,f:13,c:3,a:98.89,s:1.36,p:100.0,r:100.0,f1:100.0,t:97},{d:'Wine Quality',n:1599,f:11,c:2,a:75.67,s:3.74,p:69.29,r:69.39,f1:69.24,t:461},{d:'Yeast',n:1484,f:9,c:10,a:61.73,s:2.39,p:56.29,r:60.25,f1:56.92,t:347}].find(r => r.d === selectedDataset); return found ? found.a.toFixed(2) + '%' : 'N/A' })()}`,
                    p: `${(() => { const found = [{d:'Abalone',n:4177,f:8,c:28,a:27.12,s:2.16,p:12.04,r:12.71,f1:12.14,t:1469},{d:'Balance',n:625,f:4,c:3,a:96.48,s:1.30,p:86.67,r:96.55,f1:89.89,t:303},{d:'Banknote',n:1372,f:4,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:321},{d:'Breast Cancer',n:569,f:30,c:2,a:97.19,s:1.70,p:98.63,r:97.62,f1:98.09,t:235},{d:'Car',n:1728,f:6,c:4,a:98.32,s:0.72,p:98.82,r:97.35,f1:98.05,t:453},{d:'Dermatology',n:366,f:34,c:6,a:97.81,s:0.68,p:97.22,r:97.22,f1:96.97,t:167},{d:'Ecoli',n:336,f:8,c:8,a:95.84,s:2.16,p:97.13,r:94.29,f1:95.17,t:193},{d:'Glass',n:214,f:10,c:6,a:93.93,s:1.11,p:94.79,r:95.48,f1:94.61,t:100},{d:'Haberman',n:306,f:3,c:2,a:69.91,s:4.10,p:58.24,r:55.83,f1:56.01,t:133},{d:'Heart',n:303,f:13,c:5,a:54.79,s:4.52,p:20.91,r:24.26,f1:22.30,t:136},{d:'Ionosphere',n:351,f:34,c:2,a:94.01,s:1.91,p:93.56,r:90.89,f1:91.99,t:152},{d:'Iris',n:150,f:4,c:3,a:95.33,s:4.52,p:90.24,r:90.00,f1:89.97,t:68},{d:'Liver',n:345,f:6,c:2,a:72.17,s:7.69,p:62.44,r:62.76,f1:62.12,t:151},{d:'Mushroom',n:8124,f:22,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:1083},{d:'Optical',n:5620,f:64,c:10,a:98.81,s:0.20,p:98.69,r:98.66,f1:98.67,t:1039},{d:'Page Blocks',n:5473,f:10,c:5,a:96.80,s:0.68,p:88.53,r:86.90,f1:86.91,t:588},{d:'Sonar',n:208,f:60,c:2,a:84.13,s:1.16,p:85.29,r:85.29,f1:85.29,t:66},{d:'Spambase',n:4601,f:57,c:2,a:93.31,s:0.39,p:92.49,r:91.97,f1:92.21,t:975},{d:'Vehicle',n:846,f:18,c:5,a:79.90,s:2.71,p:64.54,r:65.35,f1:63.84,t:203},{d:'Waveform',n:5000,f:21,c:3,a:87.06,s:0.57,p:87.63,r:87.49,f1:87.43,t:556},{d:'Wine',n:178,f:13,c:3,a:98.89,s:1.36,p:100.0,r:100.0,f1:100.0,t:97},{d:'Wine Quality',n:1599,f:11,c:2,a:75.67,s:3.74,p:69.29,r:69.39,f1:69.24,t:461},{d:'Yeast',n:1484,f:9,c:10,a:61.73,s:2.39,p:56.29,r:60.25,f1:56.92,t:347}].find(r => r.d === selectedDataset); return found ? found.p.toFixed(2) + '%' : '-' })()}`,
                    r: `${(() => { const found = [{d:'Abalone',n:4177,f:8,c:28,a:27.12,s:2.16,p:12.04,r:12.71,f1:12.14,t:1469},{d:'Balance',n:625,f:4,c:3,a:96.48,s:1.30,p:86.67,r:96.55,f1:89.89,t:303},{d:'Banknote',n:1372,f:4,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:321},{d:'Breast Cancer',n:569,f:30,c:2,a:97.19,s:1.70,p:98.63,r:97.62,f1:98.09,t:235},{d:'Car',n:1728,f:6,c:4,a:98.32,s:0.72,p:98.82,r:97.35,f1:98.05,t:453},{d:'Dermatology',n:366,f:34,c:6,a:97.81,s:0.68,p:97.22,r:97.22,f1:96.97,t:167},{d:'Ecoli',n:336,f:8,c:8,a:95.84,s:2.16,p:97.13,r:94.29,f1:95.17,t:193},{d:'Glass',n:214,f:10,c:6,a:93.93,s:1.11,p:94.79,r:95.48,f1:94.61,t:100},{d:'Haberman',n:306,f:3,c:2,a:69.91,s:4.10,p:58.24,r:55.83,f1:56.01,t:133},{d:'Heart',n:303,f:13,c:5,a:54.79,s:4.52,p:20.91,r:24.26,f1:22.30,t:136},{d:'Ionosphere',n:351,f:34,c:2,a:94.01,s:1.91,p:93.56,r:90.89,f1:91.99,t:152},{d:'Iris',n:150,f:4,c:3,a:95.33,s:4.52,p:90.24,r:90.00,f1:89.97,t:68},{d:'Liver',n:345,f:6,c:2,a:72.17,s:7.69,p:62.44,r:62.76,f1:62.12,t:151},{d:'Mushroom',n:8124,f:22,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:1083},{d:'Optical',n:5620,f:64,c:10,a:98.81,s:0.20,p:98.69,r:98.66,f1:98.67,t:1039},{d:'Page Blocks',n:5473,f:10,c:5,a:96.80,s:0.68,p:88.53,r:86.90,f1:86.91,t:588},{d:'Sonar',n:208,f:60,c:2,a:84.13,s:1.16,p:85.29,r:85.29,f1:85.29,t:66},{d:'Spambase',n:4601,f:57,c:2,a:93.31,s:0.39,p:92.49,r:91.97,f1:92.21,t:975},{d:'Vehicle',n:846,f:18,c:5,a:79.90,s:2.71,p:64.54,r:65.35,f1:63.84,t:203},{d:'Waveform',n:5000,f:21,c:3,a:87.06,s:0.57,p:87.63,r:87.49,f1:87.43,t:556},{d:'Wine',n:178,f:13,c:3,a:98.89,s:1.36,p:100.0,r:100.0,f1:100.0,t:97},{d:'Wine Quality',n:1599,f:11,c:2,a:75.67,s:3.74,p:69.29,r:69.39,f1:69.24,t:461},{d:'Yeast',n:1484,f:9,c:10,a:61.73,s:2.39,p:56.29,r:60.25,f1:56.92,t:347}].find(r => r.d === selectedDataset); return found ? found.r.toFixed(2) + '%' : '-' })()}`,
                    f: `${(() => { const found = [{d:'Abalone',n:4177,f:8,c:28,a:27.12,s:2.16,p:12.04,r:12.71,f1:12.14,t:1469},{d:'Balance',n:625,f:4,c:3,a:96.48,s:1.30,p:86.67,r:96.55,f1:89.89,t:303},{d:'Banknote',n:1372,f:4,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:321},{d:'Breast Cancer',n:569,f:30,c:2,a:97.19,s:1.70,p:98.63,r:97.62,f1:98.09,t:235},{d:'Car',n:1728,f:6,c:4,a:98.32,s:0.72,p:98.82,r:97.35,f1:98.05,t:453},{d:'Dermatology',n:366,f:34,c:6,a:97.81,s:0.68,p:97.22,r:97.22,f1:96.97,t:167},{d:'Ecoli',n:336,f:8,c:8,a:95.84,s:2.16,p:97.13,r:94.29,f1:95.17,t:193},{d:'Glass',n:214,f:10,c:6,a:93.93,s:1.11,p:94.79,r:95.48,f1:94.61,t:100},{d:'Haberman',n:306,f:3,c:2,a:69.91,s:4.10,p:58.24,r:55.83,f1:56.01,t:133},{d:'Heart',n:303,f:13,c:5,a:54.79,s:4.52,p:20.91,r:24.26,f1:22.30,t:136},{d:'Ionosphere',n:351,f:34,c:2,a:94.01,s:1.91,p:93.56,r:90.89,f1:91.99,t:152},{d:'Iris',n:150,f:4,c:3,a:95.33,s:4.52,p:90.24,r:90.00,f1:89.97,t:68},{d:'Liver',n:345,f:6,c:2,a:72.17,s:7.69,p:62.44,r:62.76,f1:62.12,t:151},{d:'Mushroom',n:8124,f:22,c:2,a:100.0,s:0.00,p:100.0,r:100.0,f1:100.0,t:1083},{d:'Optical',n:5620,f:64,c:10,a:98.81,s:0.20,p:98.69,r:98.66,f1:98.67,t:1039},{d:'Page Blocks',n:5473,f:10,c:5,a:96.80,s:0.68,p:88.53,r:86.90,f1:86.91,t:588},{d:'Sonar',n:208,f:60,c:2,a:84.13,s:1.16,p:85.29,r:85.29,f1:85.29,t:66},{d:'Spambase',n:4601,f:57,c:2,a:93.31,s:0.39,p:92.49,r:91.97,f1:92.21,t:975},{d:'Vehicle',n:846,f:18,c:5,a:79.90,s:2.71,p:64.54,r:65.35,f1:63.84,t:203},{d:'Waveform',n:5000,f:21,c:3,a:87.06,s:0.57,p:87.63,r:87.49,f1:87.43,t:556},{d:'Wine',n:178,f:13,c:3,a:98.89,s:1.36,p:100.0,r:100.0,f1:100.0,t:97},{d:'Wine Quality',n:1599,f:11,c:2,a:75.67,s:3.74,p:69.29,r:69.39,f1:69.24,t:461},{d:'Yeast',n:1484,f:9,c:10,a:61.73,s:2.39,p:56.29,r:60.25,f1:56.92,t:347}].find(r => r.d === selectedDataset); return found ? found.f1.toFixed(2) + '%' : '-' })()}`,
                    protocol: '5-Fold CV',
                    isPaper: false },
                ].map((r, i) => (
                  <tr key={i} style={r.isPaper ? {} : { background: 'var(--success-light)', fontWeight: 600 }}>
                    <td>{r.label}</td>
                    <td>{r.acc}</td>
                    <td>{r.p}</td><td>{r.r}</td><td>{r.f}</td>
                    <td style={{ fontSize: '0.8rem', color: '#888' }}>{r.protocol}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="notice-box">
              <strong>Ghi chú:</strong> Kết quả published đến từ bài báo khoa học với protocol riêng (thường là 1 split hoặc K-Fold khác). JEPA+SVM dùng 5-Fold CV nhất quán. So sánh mang tính tham khảo.
            </div>
            <button className="btn" style={{ marginTop: '0.8rem', fontSize: '0.8rem', padding: '0.3rem 0.8rem' }}
              onClick={() => setSelectedDataset(null)}>Đóng</button>
          </div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>  {/* lưới 2 cột */}
        <div className="card">
          <h2>{t('dashboard.how_it_works')}</h2>
          <ul className="feature-list">
            <li>{t('dashboard.feature_jepa')}</li>
            <li>{t('dashboard.feature_svm')}</li>
            <li>{t('dashboard.feature_minmax')}</li>
            <li>{t('dashboard.feature_end_to_end')}</li>
          </ul>
        </div>

        <div className="card">  {/* thông tin hệ thống */}
          <h2>{t('dashboard.system_status')}</h2>
          <div className="status-bar status-success">
            {t('dashboard.api_running')}{health?.version}
          </div>
          <div className="metric-grid" style={{ marginTop: '0.8rem' }}>
            <div className="metric-card">
              <div className="metric-value">{apiInfo?.jepa_config?.epochs || '-'}</div>
              <div className="metric-label">{t('dashboard.jepa_epochs')}</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{apiInfo?.svm_config?.kernel || '-'}</div>
              <div className="metric-label">{t('dashboard.svm_kernel')}</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{apiInfo?.jepa_config?.learning_rate || '-'}</div>
              <div className="metric-label">{t('dashboard.learning_rate')}</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">
                <span className={`badge ${ensembleInfo?.info?.model_trained ? 'badge-success' : 'badge-danger'}`}>
                  {ensembleInfo?.info?.model_trained ? t('dashboard.trained') : t('dashboard.not_trained')}
                </span>
              </div>
              <div className="metric-label">{t('dashboard.model_status')}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '0' }}>  {/* danh sách công nghệ */}
        <h2>{t('dashboard.technologies')}</h2>
        <div className="tech-stack">
          {[
            { name: 'FastAPI', desc: t('dashboard.backend_api') },
            { name: 'PyTorch', desc: t('dashboard.jepa_model') },
            { name: 'scikit-learn', desc: t('dashboard.svm_grid') },
            { name: 'React', desc: t('dashboard.frontend_ui') },
            { name: 'Python', desc: t('dashboard.core_lang') },
          ].map((tech) => (
            <div key={tech.name} className="tech-item">
              <div className="tech-name">{tech.name}</div>
              <div className="tech-desc">{tech.desc}</div>
            </div>
          ))}
         </div>
       </div>

      <div className="card">  {/* sơ đồ kiến trúc */}
        <h2>{t('dashboard.about_title')}</h2>
        <p style={{ lineHeight: 1.7, marginBottom: '1rem' }}>{t('dashboard.about_desc')}</p>

        <style>{`
          .flowchart { display: flex; flex-direction: column; align-items: center; gap: 0.5rem; padding: 1rem; background: #f8f9ff; border-radius: 12px; margin: 0.5rem 0; }
          .fc-row { display: flex; align-items: center; gap: 0.5rem; }
          .fc-split { display: flex; gap: 2rem; }
          .fc-branch { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; }
          .fc-box { padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.85rem; font-weight: 600; text-align: center; min-width: 90px; }
          .fc-box.input { background: #e3f2fd; color: #1565c0; border: 2px solid #1565c0; }
          .fc-box.process { background: #fff3e0; color: #e65100; border: 2px solid #e65100; }
          .fc-box.model { background: #f3e5f5; color: #7b1fa2; border: 2px solid #7b1fa2; }
          .fc-box.output { background: #e8f5e9; color: #2e7d32; border: 2px solid #2e7d32; }
          .fc-box.final { background: #e8eaf6; color: #283593; border: 2px solid #283593; }
          .fc-box.result { background: #fff8e1; color: #f57f17; border: 2px solid #f57f17; }
          .fc-arrow { color: #999; font-size: 1.3rem; font-weight: 700; }
          .fc-arrow-down { color: #999; font-size: 1rem; }
          .fc-arrow-up { color: #999; font-size: 1rem; transform: rotate(180deg); }
        `}</style>

        <div className="flowchart">
          <div className="fc-row">
            <div className="fc-box input">{t('dashboard.fc_input')}</div>
            <div className="fc-arrow">→</div>
            <div className="fc-box process">{t('dashboard.fc_scale')}</div>
            <div className="fc-arrow">→</div>
            <div className="fc-box model">{t('dashboard.fc_jepa_ssl')}</div>
            <div className="fc-arrow">→</div>
            <div className="fc-box process">{t('dashboard.fc_jepa')}</div>
            <div className="fc-arrow">→</div>
            <div className="fc-box output">{t('dashboard.fc_embed')}</div>
            <div className="fc-arrow">→</div>
            <div className="fc-box model">{t('dashboard.fc_svm')}</div>
            <div className="fc-arrow">→</div>
            <div className="fc-box result">{t('dashboard.fc_result')}</div>
          </div>
        </div>

        <div className="info-box" style={{ textAlign: 'center' }}>
          {t('dashboard.about_legend')}
        </div>

        <h3 style={{ marginTop: '1.8rem', color: 'var(--primary)', borderBottom: '2px solid var(--primary-light)', paddingBottom: '0.5rem' }}>
          JEPA+SVM — Sơ đồ kiến trúc tổng thể
        </h3>

        <style>{`
          .arch { display: flex; flex-direction: column; gap: 1rem; padding: 0.5rem; }
          .arch-phase { border-radius: 10px; padding: 0.8rem; border: 2px solid #e0e0e0; }
          .arch-phase-title { font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; padding: 0.2rem 0.6rem; border-radius: 6px; display: inline-block; }
          .arch-row { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; justify-content: center; }
          .arch-box { padding: 0.4rem 0.7rem; border-radius: 6px; font-size: 0.78rem; font-weight: 600; text-align: center; }
          .arch-box.data { background: #e3f2fd; color: #1565c0; border: 1.5px solid #1565c0; }
          .arch-box.preprocess { background: #fff3e0; color: #e65100; border: 1.5px solid #e65100; }
          .arch-box.jepa-core { background: #f3e5f5; color: #7b1fa2; border: 1.5px solid #7b1fa2; }
          .arch-box.embed { background: #e8f5e9; color: #2e7d32; border: 1.5px solid #2e7d32; }
          .arch-box.svm { background: #fce4ec; color: #c62828; border: 1.5px solid #c62828; }
          .arch-box.eval { background: #fff8e1; color: #f57f17; border: 1.5px solid #f57f17; }
          .arch-arrow { color: #999; font-size: 1rem; font-weight: 700; }
          .arch-sub { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; padding: 0.4rem; background: #fafafa; border-radius: 6px; margin: 0.2rem; min-width: 120px; }
          .arch-sub-box { padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.72rem; text-align: center; width: 100%; }
          .arch-sub-box.enc { background: #ede7f6; color: #4527a0; border: 1px solid #4527a0; }
          .arch-sub-box.tgt { background: #fce4ec; color: #c62828; border: 1px solid #c62828; }
          .arch-sub-box.pred { background: #fff3e0; color: #e65100; border: 1px solid #e65100; }
          .arch-sub-box.loss { background: #fff8e1; color: #f57f17; border: 1px dashed #f57f17; }
          .arch-sub-box.head { background: #e8f5e9; color: #2e7d32; border: 1px solid #2e7d32; }
          .arch-sub-box.svm-item { background: #fce4ec; color: #c62828; border: 1px solid #c62828; }
          .arch-grid { display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center; }
          .arch-tag { font-size: 0.7rem; color: #888; font-weight: 400; }
          .arch-fold-row { display: flex; gap: 0.3rem; align-items: center; justify-content: center; flex-wrap: wrap; }
          .arch-fold { padding: 0.3rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; background: #e8eaf6; color: #283593; border: 1px solid #283593; }
        `}</style>

        <div className="arch">

          {/* ===== PHA 1: TIỀN XỬ LÝ ===== */}
          <div className="arch-phase" style={{borderColor:'#e65100'}}>
            <div className="arch-phase-title" style={{background:'#fff3e0',color:'#e65100'}}>PHA 1 — TIỀN XỬ LÝ DỮ LIỆU</div>
            <div className="arch-row">
              <div className="arch-box data">CSV Dataset<br/><span className="arch-tag">features + target</span></div>
              <div className="arch-arrow">→</div>
              <div className="arch-box preprocess">OneHotEncode<br/><span className="arch-tag">categorical → numeric</span></div>
              <div className="arch-arrow">→</div>
              <div className="arch-box preprocess">MinMaxScaler<br/><span className="arch-tag">scale về [-1, 1]</span></div>
              <div className="arch-arrow">→</div>
              <div className="arch-box data">X_scaled<br/><span className="arch-tag">shape (N, features)</span></div>
            </div>
          </div>

          {/* ===== PHA 2: JEPA SELF-SUPERVISED ===== */}
          <div className="arch-phase" style={{borderColor:'#7b1fa2'}}>
            <div className="arch-phase-title" style={{background:'#f3e5f5',color:'#7b1fa2'}}>PHA 2 — JEPA SELF-SUPERVISED (không cần nhãn)</div>
            <div className="arch-row" style={{marginBottom:'0.5rem'}}>
              <div className="arch-box data">X_scaled</div>
              <div className="arch-arrow">→</div>
              <div className="arch-box preprocess">Random Mask (30%)<br/><span className="arch-tag">context = x × mask</span></div>
              <div className="arch-arrow">→</div>
              <div className="arch-box preprocess">Original x<br/><span className="arch-tag">target = x (no mask)</span></div>
            </div>
            <div className="arch-grid">
              <div className="arch-sub">
                <div className="arch-box jepa-core" style={{width:'100%'}}>Context Path</div>
                <div className="arch-sub-box enc">Encoder MLP<br/><span className="arch-tag">Linear→BN→ReLU→...→Linear</span></div>
                <div className="arch-arrow" style={{fontSize:'0.8rem'}}>↓</div>
                <div className="arch-sub-box pred">Predictor MLP<br/><span className="arch-tag">Linear→BN→ReLU→Linear</span></div>
                <div className="arch-arrow" style={{fontSize:'0.8rem'}}>↓</div>
                <div className="arch-sub-box head" style={{borderStyle:'dashed'}}>pred_emb (32-dim)</div>
              </div>
              <div style={{display:'flex',alignItems:'center',fontSize:'1.5rem',color:'#999'}}>✖</div>
              <div className="arch-sub">
                <div className="arch-box jepa-core" style={{width:'100%'}}>Target Path</div>
                <div className="arch-sub-box tgt">Target Encoder MLP<br/><span className="arch-tag">giống Encoder, EMA update</span></div>
                <div className="arch-arrow" style={{fontSize:'0.8rem'}}>↓</div>
                <div className="arch-sub-box head" style={{borderStyle:'dashed',background:'#f1f8e9',borderColor:'#558b2f'}}>tgt_emb (32-dim)</div>
                <div className="arch-tag" style={{marginTop:'0.2rem'}}>stop-gradient (detach)</div>
              </div>
            </div>
            <div className="arch-row" style={{marginTop:'0.5rem'}}>
              <div className="arch-sub-box loss" style={{minWidth:250}}>
                <strong>MSE Loss = ||normalize(pred_emb) − normalize(tgt_emb)||²</strong><br/>
                <span className="arch-tag">Backprop → Encoder + Predictor</span>
              </div>
            </div>
            <div className="arch-row" style={{marginTop:'0.3rem'}}>
              <div className="arch-box eval" style={{fontSize:'0.72rem',minWidth:250}}>
                <strong>EMA Update:</strong> θ_target = 0.995 × θ_target + 0.005 × θ_encoder<br/>
                <span className="arch-tag">Target Encoder không có gradient, cập nhật bằng trung bình động</span>
              </div>
            </div>
          </div>

          {/* ===== PHA 3: SUPERVISED FINE-TUNE ===== */}
          <div className="arch-phase" style={{borderColor:'#2e7d32'}}>
            <div className="arch-phase-title" style={{background:'#e8f5e9',color:'#2e7d32'}}>PHA 3 — SUPERVISED FINE-TUNE (có nhãn)</div>
            <div className="arch-row">
              <div className="arch-box data">X_scaled + y</div>
              <div className="arch-arrow">→</div>
              <div className="arch-box jepa-core">Encoder<br/><span className="arch-tag">(có gradient)</span></div>
              <div className="arch-arrow">→</div>
              <div className="arch-box embed">emb (32-dim)</div>
            </div>
            <div className="arch-grid" style={{marginTop:'0.3rem'}}>
              <div className="arch-sub" style={{minWidth:'160px'}}>
                <div className="arch-sub-box head">Classification Head<br/><span className="arch-tag">Linear(32, n_classes)</span></div>
                <div className="arch-arrow" style={{fontSize:'0.8rem'}}>↓</div>
                <div className="arch-box eval" style={{fontSize:'0.72rem'}}>
                  <strong>CrossEntropy Loss</strong><br/>
                  <span className="arch-tag">CE(logits, y_true)</span>
                </div>
              </div>
              <div style={{display:'flex',alignItems:'center',color:'#999'}}>+</div>
              <div className="arch-sub" style={{minWidth:'160px'}}>
                <div className="arch-sub-box loss" style={{background:'#f1f8e9',borderColor:'#558b2f'}}>
                  <strong>Contrastive Loss</strong><br/>
                  <span className="arch-tag">kéo cùng lớp gần nhau<br/>đẩy khác lớp xa nhau</span>
                </div>
              </div>
            </div>
            <div className="arch-row" style={{marginTop:'0.3rem'}}>
              <div className="arch-box eval" style={{minWidth:300,fontSize:'0.72rem'}}>
                <strong>Tổng loss</strong> = CrossEntropy + 0.1 × Contrastive<br/>
                <span className="arch-tag">Backprop → Encoder + Classification Head</span>
              </div>
            </div>
          </div>

          {/* ===== PHA 4: TRÍCH XUẤT EMBEDDING ===== */}
          <div className="arch-phase" style={{borderColor:'#2e7d32'}}>
            <div className="arch-phase-title" style={{background:'#e8f5e9',color:'#2e7d32'}}>PHA 4 — TRÍCH XUẤT EMBEDDINGS</div>
            <div className="arch-row">
              <div className="arch-box data">X_train (scaled)</div>
              <div className="arch-arrow">→</div>
              <div className="arch-box jepa-core">Encoder (trained)</div>
              <div className="arch-arrow">→</div>
              <div className="arch-box embed">train_emb<br/><span className="arch-tag">shape (N_train, 32)</span></div>
            </div>
            <div className="arch-row" style={{marginTop:'0.2rem'}}>
              <div className="arch-box data">X_test (scaled)</div>
              <div className="arch-arrow">→</div>
              <div className="arch-box jepa-core">Encoder (trained)</div>
              <div className="arch-arrow">→</div>
              <div className="arch-box embed">test_emb<br/><span className="arch-tag">shape (N_test, 32)</span></div>
            </div>
          </div>

          {/* ===== PHA 5: SVM ===== */}
          <div className="arch-phase" style={{borderColor:'#c62828'}}>
            <div className="arch-phase-title" style={{background:'#fce4ec',color:'#c62828'}}>PHA 5 — SVM CLASSIFICATION</div>
            <div className="arch-row">
              <div className="arch-box svm">GridSearchCV<br/><span className="arch-tag">C=[0.1,1,10,100]<br/>gamma=['scale','auto',0.01,0.1]</span></div>
              <div className="arch-arrow">→</div>
              <div className="arch-box svm">Best SVC(RBF)<br/><span className="arch-tag">C* + gamma* tối ưu</span></div>
              <div className="arch-arrow">→</div>
              <div className="arch-box svm">SVM.train(train_emb, y_train)</div>
            </div>
            <div className="arch-row" style={{marginTop:'0.3rem'}}>
              <div className="arch-box svm">SVM.predict(test_emb)</div>
              <div className="arch-arrow">→</div>
              <div className="arch-box svm">y_pred</div>
              <div className="arch-arrow">→</div>
              <div className="arch-box eval"><strong>Metrics</strong><br/><span className="arch-tag">Accuracy, Precision, Recall, F1, Confusion Matrix</span></div>
            </div>
          </div>

          {/* ===== PHA 6: K-FOLD CV ===== */}
          <div className="arch-phase" style={{borderColor:'#f57f17'}}>
            <div className="arch-phase-title" style={{background:'#fff8e1',color:'#f57f17'}}>PHA 6 — K-FOLD CROSS VALIDATION (lặp lại Pha 1→5 với 5 fold)</div>
            <div className="arch-fold-row">
              <div className="arch-fold">Fold 1: train → test</div>
              <div className="arch-arrow" style={{fontSize:'0.8rem'}}>→</div>
              <div className="arch-fold">Acc₁</div>
              <div style={{margin:'0 0.5rem'}}></div>
              <div className="arch-fold">Fold 2: train → test</div>
              <div className="arch-arrow" style={{fontSize:'0.8rem'}}>→</div>
              <div className="arch-fold">Acc₂</div>
              <div style={{margin:'0 0.5rem'}}></div>
              <div className="arch-fold">Fold 3: train → test</div>
              <div className="arch-arrow" style={{fontSize:'0.8rem'}}>→</div>
              <div className="arch-fold">Acc₃</div>
              <div style={{margin:'0 0.5rem'}}></div>
              <div className="arch-fold">Fold 4: train → test</div>
              <div className="arch-arrow" style={{fontSize:'0.8rem'}}>→</div>
              <div className="arch-fold">Acc₄</div>
              <div style={{margin:'0 0.5rem'}}></div>
              <div className="arch-fold">Fold 5: train → test</div>
              <div className="arch-arrow" style={{fontSize:'0.8rem'}}>→</div>
              <div className="arch-fold">Acc₅</div>
            </div>
            <div className="arch-row" style={{marginTop:'0.5rem'}}>
              <div className="arch-box eval" style={{minWidth:300}}>
                <strong>Kết quả cuối:</strong> Accuracy = mean(Acc₁…Acc₅)  ±  std(Acc₁…Acc₅)
              </div>
            </div>
          </div>

        </div>
      </div>
    </>
  )
}
