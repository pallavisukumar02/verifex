import React, { useEffect, useMemo, useState } from 'react'
import { getLiveTransactions } from '../api'

const verdictStyle = {
  LEGIT: 'badge-legit',
  REVIEW: 'badge-review',
  FRAUD: 'badge-fraud',
}

const FALLBACK_USERS = [
  'user_23',
  'user_36',
  'user_14',
  'user_26',
  'user_2',
  'user_11',
  'user_48',
]

const FALLBACK_LOCATIONS = ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad']
const FALLBACK_MERCHANTS = ['Amazon', 'Swiggy', 'Zomato', 'Flipkart', 'Netflix']

function sampleFallback() {
  const amount = Math.floor(100 + Math.random() * 99000)
  const verdict = Math.random() > 0.82 ? 'FRAUD' : Math.random() > 0.7 ? 'REVIEW' : 'LEGIT'
  return {
    id: Date.now() + Math.random(),
    user_id: FALLBACK_USERS[Math.floor(Math.random() * FALLBACK_USERS.length)],
    amount,
    time: `${String(Math.floor(Math.random() * 24)).padStart(2, '0')}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`,
    location: FALLBACK_LOCATIONS[Math.floor(Math.random() * FALLBACK_LOCATIONS.length)],
    merchant: FALLBACK_MERCHANTS[Math.floor(Math.random() * FALLBACK_MERCHANTS.length)],
    verdict,
  }
}

export default function TransactionFeed() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true

    async function refreshFeed() {
      try {
        const data = await getLiveTransactions()
        if (!mounted) return
        setItems(data.map((item, idx) => ({ id: item.user_id + '-' + idx, ...item })))
      } catch (err) {
        if (!mounted) return
        setItems((prev) => [sampleFallback(), ...prev].slice(0, 8))
      } finally {
        if (mounted) setLoading(false)
      }
    }

    refreshFeed()
    const timer = setInterval(refreshFeed, 4500)
    return () => {
      mounted = false
      clearInterval(timer)
    }
  }, [])

  const summary = useMemo(() => {
    return items.reduce((acc, item) => {
      acc[item.verdict] = (acc[item.verdict] || 0) + 1
      return acc
    }, {})
  }, [items])

  return (
    <div className="transaction-panel">
      <div className="panel-head">
        <span className="panel-title">Live Transaction Stream</span>
        <span className="panel-sub">Powered by dataset sampling</span>
      </div>
      <div className="badge-legend live-legend">
        <span><strong>{summary.LEGIT || 0}</strong> legit</span>
        <span><strong>{summary.REVIEW || 0}</strong> review</span>
        <span><strong>{summary.FRAUD || 0}</strong> fraud</span>
      </div>
      <div className="transaction-list">
        {loading ? (
          Array.from({ length: 5 }).map((_, idx) => (
            <div className="transaction-item skeleton" key={idx} />
          ))
        ) : items.length === 0 ? (
          <div className="empty"><span style={{ fontSize: 13 }}>No live transactions available</span></div>
        ) : (
          items.map((item, idx) => (
            <div className="transaction-item" key={item.id} style={{ animationDelay: `${idx * 60}ms` }}>
              <div>
                <div className={`badge-pill ${verdictStyle[item.verdict]}`}>{item.verdict}</div>
                <p><strong>{item.user_id}</strong> · {item.merchant}</p>
                <div className="txn-mini">{item.location} · {item.device_id || 'device'} · {item.time}</div>
              </div>
              <div className="txn-right">
                <span className={`txn-amt ${item.verdict === 'FRAUD' ? 'fraud' : 'legit'}`}>₹{Number(item.amount).toLocaleString('en-IN')}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
