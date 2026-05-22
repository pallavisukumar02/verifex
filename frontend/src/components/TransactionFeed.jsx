import React, { useEffect, useMemo, useState } from 'react'

const BASE_FEED = [
  { id: 1, user: 'user_23', amount: 21213, time: '03:17', verdict: 'LEGIT' },
  { id: 2, user: 'user_36', amount: 44893, time: '03:17', verdict: 'LEGIT' },
  { id: 3, user: 'user_14', amount: 64052, time: '03:17', verdict: 'FRAUD' },
  { id: 4, user: 'user_26', amount: 40481, time: '03:17', verdict: 'LEGIT' },
  { id: 5, user: 'user_2', amount: 89855, time: '03:17', verdict: 'LEGIT' },
]

const verdictStyle = {
  LEGIT: 'badge-legit',
  REVIEW: 'badge-review',
  FRAUD: 'badge-fraud',
}

export default function TransactionFeed() {
  const [items, setItems] = useState(BASE_FEED)

  useEffect(() => {
    const ticker = setInterval(() => {
      const next = {
        id: Date.now(),
        user: `user_${Math.floor(Math.random() * 60)}`,
        amount: Math.floor(100 + Math.random() * 99000),
        time: `${String(Math.floor(Math.random() * 24)).padStart(2, '0')}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`,
        verdict: Math.random() > 0.85 ? 'FRAUD' : Math.random() > 0.7 ? 'REVIEW' : 'LEGIT',
      }
      setItems(prev => [next, ...prev].slice(0, 6))
    }, 3500)
    return () => clearInterval(ticker)
  }, [])

  const summary = useMemo(() => {
    const count = items.reduce((acc, item) => {
      acc[item.verdict] = (acc[item.verdict] || 0) + 1
      return acc
    }, {})
    return count
  }, [items])

  return (
    <div>
      <div className="section-title">Live Feed</div>
      <div className="badge-legend">
        <span><strong>{summary.LEGIT || 0}</strong> legitimate</span>
        <span><strong>{summary.REVIEW || 0}</strong> review</span>
        <span><strong>{summary.FRAUD || 0}</strong> fraud</span>
      </div>
      <div className="transaction-list">
        {items.map((item) => (
          <div className="transaction-item" key={item.id}>
            <div>
              <div className={`badge-pill ${verdictStyle[item.verdict]}`}>{item.verdict}</div>
              <p><strong>{item.user}</strong> — ₹{item.amount.toLocaleString()}</p>
            </div>
            <time>{item.time}</time>
          </div>
        ))}
      </div>
    </div>
  )
}
