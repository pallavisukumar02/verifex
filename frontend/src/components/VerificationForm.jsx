import React, { useState } from 'react'
import { scoreTransaction } from '../api'
import ExplanationPanel from './ExplanationPanel'

const INITIAL_PAYLOAD = {
  user_id: 'user_1',
  amount: 1000,
  location: 'Bengaluru',
  device_id: 'phone_1',
  time: '12:00',
}

export default function VerificationForm(){
  const [payload, setPayload] = useState(INITIAL_PAYLOAD)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  function updateField(field, value){
    setPayload(prev=>({ ...prev, [field]: value }))
  }

  async function onSubmit(e){
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try{
      const res = await scoreTransaction(payload)
      setResult(res)
    }catch(err){
      setError(err?.response?.data?.detail || err?.message || 'Unable to score transaction')
    }finally{
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="section-title">Verify Transaction</div>
      <form className="form-grid" onSubmit={onSubmit}>
        <label>
          User ID
          <input value={payload.user_id} onChange={e=>updateField('user_id', e.target.value)} />
        </label>
        <label>
          Amount (₹)
          <input type="number" value={payload.amount} onChange={e=>updateField('amount', Number(e.target.value))} />
        </label>
        <label>
          Location
          <input value={payload.location} onChange={e=>updateField('location', e.target.value)} />
        </label>
        <label>
          Device ID
          <input value={payload.device_id} onChange={e=>updateField('device_id', e.target.value)} />
        </label>
        <label>
          Time (HH:MM)
          <input value={payload.time} onChange={e=>updateField('time', e.target.value)} />
        </label>
        <div style={{ display:'flex', gap: 12, alignItems:'center', marginTop: 6 }}>
          <button type="submit" disabled={loading}>{loading ? 'Scoring...' : 'Score Transaction'}</button>
          <button type="button" onClick={()=>{ setPayload(INITIAL_PAYLOAD); setResult(null); setError(null) }} style={{ background:'#0f172a' }}>Reset</button>
        </div>
      </form>

      {error && (
        <div className="result-card" style={{ borderColor:'#fecaca', background:'#fef2f2', color:'#991b1b' }}>
          {error}
        </div>
      )}

      {result && <ExplanationPanel result={result} />}
    </div>
  )
}
