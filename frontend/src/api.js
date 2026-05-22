import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export async function scoreTransaction(payload){
  const resp = await axios.post(`${API_BASE}/score-transaction`, payload)
  return resp.data
}
