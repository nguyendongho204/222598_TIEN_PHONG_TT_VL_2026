import { useTranslation } from 'react-i18next'  // đa ngôn ngữ

const ENDPOINTS = [  // danh sách các API endpoint
  {
    method: 'GET', path: '/api/health', desc_key: 'api.health_desc',
    body: null, response: '{ "status": "ok", "version": "1.0" }',
  },
  {
    method: 'GET', path: '/api/info', desc_key: 'api.info_desc',
    body: null, response: '{ "jepa_config": {...}, "svm_config": {...} }',
  },
  {
    method: 'POST', path: '/api/train-dataset', desc_key: 'api.train_desc',
    body: 'file (CSV), test_size, num_runs',
    response: '{ "status": "success", "accuracy": 0.95, "accuracy_std": 0.02, ... }',
  },
  {
    method: 'GET', path: '/api/info', desc_key: 'api.ens_info_desc',
    body: null, response: '{ "model_type": "JEPA+SVM", "model_trained": true }',
  },
  {
    method: 'POST', path: '/api/ensemble/predict', desc_key: 'api.predict_desc',
    body: '{ "features": [[1.0, 2.0, ...]] }',
    response: '{ "predictions": [0, 1], "confidences": [0.95, 0.87] }',
  },
  {
    method: 'GET', path: '/api/history/all', desc_key: 'api.history_desc',
    body: null, response: '{ "histories": [...] }',
  },
]

export default function ApiDoc() {
  const { t } = useTranslation()  // hàm dịch

  return (
    <>
      <div className="card">
        <h2>{t('api.title')}</h2>
        <p style={{ color: '#888', marginBottom: '1rem' }}>{t('api.desc')}</p>

        <div style={{ overflowX: 'auto' }}>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>{t('api.col_endpoint')}</th>
                <th>{t('api.col_desc')}</th>
                <th>{t('api.col_request')}</th>
                <th>{t('api.col_response')}</th>
              </tr>
            </thead>
            <tbody>
              {ENDPOINTS.map((ep, i) => (
                <tr key={i}>
                  <td>
                    <span className={`badge ${ep.method === 'GET' ? 'badge-success' : 'badge-info'}`}>
                      {ep.method}
                    </span>
                  </td>
                  <td><code style={{ fontSize: '0.85rem' }}>{ep.path}</code></td>
                  <td style={{ fontSize: '0.85rem' }}>{t(ep.desc_key)}</td>
                  <td><code style={{ fontSize: '0.8rem', wordBreak: 'break-all' }}>{ep.body || '—'}</code></td>
                  <td><code style={{ fontSize: '0.8rem', wordBreak: 'break-all' }}>{ep.response}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: '1.5rem' }}>
          <h3 style={{ marginBottom: '0.5rem' }}>{t('api.note_title')}</h3>
          <ul className="feature-list">
            <li>{t('api.note1')}</li>
            <li>{t('api.note2')}</li>
            <li>{t('api.note3')}</li>
          </ul>
        </div>
      </div>
    </>
  )
}
