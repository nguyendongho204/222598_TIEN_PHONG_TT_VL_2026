import i18n from 'i18next'  // thư viện i18n
import { initReactI18next } from 'react-i18next'  // tích hợp với React
import en from './locales/en.json'  // file ngôn ngữ tiếng Anh
import vi from './locales/vi.json'  // file ngôn ngữ tiếng Việt

i18n.use(initReactI18next).init({  // khởi tạo i18n
  resources: { en: { translation: en }, vi: { translation: vi } },  // tài nguyên ngôn ngữ
  lng: localStorage.getItem('lang') || 'vi',  // ngôn ngữ mặc định từ localStorage hoặc 'vi'
  fallbackLng: 'en',  // ngôn ngữ dự phòng
  interpolation: { escapeValue: false },  // không escape giá trị
})

export default i18n  // xuất đối tượng i18n
