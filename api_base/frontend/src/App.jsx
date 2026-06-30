import { Routes, Route } from 'react-router-dom'  // định tuyến trang
import Navbar from './components/Navbar'  // thanh điều hướng
import Dashboard from './pages/Dashboard'  // trang tổng quan
import Training from './pages/Training'  // trang huấn luyện
import Results from './pages/Results'  // trang kết quả
import ApiDoc from './pages/ApiDoc'  // trang tài liệu API
import History from './pages/History'  // trang lịch sử

export default function App() {
  return (
    <>
      <Navbar />  {/* thanh điều hướng toàn cục */}
      <div className="main-content">  {/* nội dung chính */}
        <Routes>  {/* định nghĩa các route */}
          <Route path="/" element={<Dashboard />} />
          <Route path="/training" element={<Training />} />
          <Route path="/results" element={<Results />} />
          <Route path="/api-doc" element={<ApiDoc />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </div>
    </>
  )
}
