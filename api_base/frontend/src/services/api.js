const BASE_URL = '/api'  // URL gốc của backend

async function handleResponse(res) {  // xử lý response HTTP
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || err.message || 'Request failed')
  }
  return res.json()
}

export async function healthCheck() {  // kiểm tra trạng thái server
  const res = await fetch(`${BASE_URL}/health`)
  return handleResponse(res)
}

export async function getApiInfo() {  // lấy thông tin cấu hình API
  const res = await fetch(`${BASE_URL}/info`)
  return handleResponse(res)
}

export async function uploadFile(file) {  // tải file lên server
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${BASE_URL}/files/upload`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(res)
}

export async function trainDataset(file, params) {  // gửi file và tham số để huấn luyện
  const formData = new FormData()
  formData.append('file', file)
  formData.append('test_size', String(params.test_size))
  formData.append('num_runs', String(params.num_runs ?? 1))
  const res = await fetch(`${BASE_URL}/train-dataset`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(res)
}

export async function getAllHistories() {  // lấy tất cả lịch sử huấn luyện
  const res = await fetch(`${BASE_URL}/history/all`)
  return handleResponse(res)
}

export async function getHistory(id) {  // lấy chi tiết một lịch sử theo id
  const res = await fetch(`${BASE_URL}/history/${id}`)
  return handleResponse(res)
}

export async function deleteHistory(id) {  // xoá một bản ghi lịch sử
  const res = await fetch(`${BASE_URL}/history/${id}`, { method: 'DELETE' })
  return handleResponse(res)
}

export async function getEnsembleInfo() {  // lấy thông tin ensemble model
  const res = await fetch(`${BASE_URL}/ensemble/info`)
  return handleResponse(res)
}
