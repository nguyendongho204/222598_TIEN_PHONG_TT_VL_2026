import { useState, useEffect } from 'react'  // hook trạng thái và lifecycle
import { NavLink } from 'react-router-dom'  // điều hướng với active style
import { useTranslation } from 'react-i18next'  // đa ngôn ngữ
import './Navbar.css'

const TRAINING_STORAGE_KEY = 'trainingResult'  // key localStorage cho kết quả training

export default function Navbar() {
  const { t, i18n } = useTranslation()  // hàm dịch và đối tượng i18n
  const [dark, setDark] = useState(() => {  // trạng thái theme tối/sáng
    return localStorage.getItem('theme') === 'dark'
  })
  const [trainingStatus, setTrainingStatus] = useState(null)  // trạng thái training (running/complete/null)

  const toggleLang = () => {  // chuyển đổi ngôn ngữ
    const next = i18n.language === 'vi' ? 'en' : 'vi'
    i18n.changeLanguage(next)
    localStorage.setItem('lang', next)
  }

  useEffect(() => {  // cập nhật data-theme khi dark thay đổi
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  useEffect(() => {  // kiểm tra trạng thái training theo chu kỳ
    const check = () => {
      try {
        const stored = localStorage.getItem(TRAINING_STORAGE_KEY)
        if (stored) {
          const parsed = JSON.parse(stored)
          setTrainingStatus(parsed.status)
        } else {
          setTrainingStatus(null)
        }
      } catch { setTrainingStatus(null) }
    }
    check()
    const interval = setInterval(check, 3000)  // kiểm tra mỗi 3 giây
    window.addEventListener('storage', check)  // lắng nghe sự kiện storage
    return () => {
      clearInterval(interval)
      window.removeEventListener('storage', check)
    }
  }, [])

  return (
    <nav className="navbar">  {/* thanh điều hướng chính */}
      <div className="navbar-brand">  {/* logo + badge training */}
        <span className="navbar-logo">JEPA+SVM</span>
        {trainingStatus === 'running' && (
          <span className="training-badge">
            <span className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
            {t('nav.training_badge')}
          </span>
        )}
        {trainingStatus === 'complete' && (
          <span className="training-badge done">{t('nav.done_badge')}</span>
        )}
      </div>
      <div className="navbar-links">  {/* các link điều hướng */}
        <NavLink to="/" end className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          {t('nav.dashboard')}
        </NavLink>
        <NavLink to="/training" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          {t('nav.training')}
        </NavLink>
        <NavLink to="/results" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          {t('nav.results')}
        </NavLink>
        <NavLink to="/api-doc" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          API
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          {t('nav.history')}
        </NavLink>
      </div>
      <div className="navbar-actions">
        <button className="theme-toggle" onClick={toggleLang} title={i18n.language === 'vi' ? 'English' : 'Tiếng Việt'}>
          {i18n.language === 'vi' ? 'EN' : 'VI'}
        </button>
        <button className="theme-toggle" onClick={() => setDark((d) => !d)} title={t('nav.toggle_theme')}>
        {dark ? (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5"/>
            <line x1="12" y1="1" x2="12" y2="3"/>
            <line x1="12" y1="21" x2="12" y2="23"/>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
            <line x1="1" y1="12" x2="3" y2="12"/>
            <line x1="21" y1="12" x2="23" y2="12"/>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
          </svg>
        )}
      </button>
      </div>
    </nav>
  )
}
