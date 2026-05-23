import React, { useEffect, useMemo, useState } from 'react'
import { scoreTransaction as apiScoreTransaction } from './api'
import TransactionFeed from './components/TransactionFeed'
import './styles.css'

const DEFAULT_ACCOUNT = {
  fname: '',
  lname: '',
  email: '',
  phone: '',
  bank: 'SBI',
  acctype: 'Savings',
  avgspend: '10000',
  city: '',
  threshold: 'medium',
}

const DEFAULT_TEST = {
  amount: 85000,
  hour: 3,
  location: 'Mumbai',
  device: 'known_device',
  merchant: 'Amazon',
  note: '',
}

function formatInitials(name = '', surname = '') {
  return `${name[0] || '?'}${surname[0] || ''}`.toUpperCase() || '?'
}

function localScore(tx, avgspend) {
  const newDev = /new|unknown/i.test(tx.device)
  const night = tx.hour < 5 || tx.hour > 22
  const big = tx.amount > (Number(avgspend) || 5000) * 2
  const foreign = ['Unknown', 'Foreign', 'Abroad'].includes(tx.location)
  let s = 0
  if (newDev) s += 0.38
  if (night) s += 0.25
  if (big) s += 0.22
  if (foreign) s += 0.18
  s = Math.min(0.97, s + Math.random() * 0.05)
  const reasons = []
  if (newDev) reasons.push({ label: 'Unknown device detected', score: 0.38, type: 'risk' })
  if (night) reasons.push({ label: `Transaction at ${tx.hour}:00`, score: 0.25, type: 'risk' })
  if (big) reasons.push({ label: 'Amount above your average', score: 0.22, type: 'risk' })
  if (foreign) reasons.push({ label: 'Unusual location', score: 0.18, type: 'risk' })
  if (!reasons.length) {
    reasons.push({ label: 'Familiar device', score: -0.25, type: 'safe' })
    reasons.push({ label: 'Normal amount', score: -0.2, type: 'safe' })
    reasons.push({ label: 'Known location', score: -0.15, type: 'safe' })
  }
  return { score: Number(s.toFixed(3)), verdict: s > 0.5 ? 'FRAUD' : 'LEGIT', reasons }
}

export default function App() {
  const storedAccount = JSON.parse(localStorage.getItem('fv_account') || 'null')
  const storedHistory = JSON.parse(localStorage.getItem('fv_history') || '[]')
  const storedAlerts = JSON.parse(localStorage.getItem('fv_alerts') || '[]')

  const [tab, setTab] = useState(storedAccount ? 'profile' : 'setup')
  const [account, setAccount] = useState(storedAccount || DEFAULT_ACCOUNT)
  const [form, setForm] = useState(storedAccount || DEFAULT_ACCOUNT)
  const [history, setHistory] = useState(storedHistory)
  const [alerts, setAlerts] = useState(storedAlerts)
  const [testData, setTestData] = useState(DEFAULT_TEST)
  const [result, setResult] = useState(null)
  const [resultFlipped, setResultFlipped] = useState(false)
  const [toastVisible, setToastVisible] = useState(false)
  const [loading, setLoading] = useState(false)

  const fraudTxns = useMemo(() => history.filter((item) => item.verdict === 'FRAUD'), [history])
  const safeTxns = useMemo(() => history.filter((item) => item.verdict === 'LEGIT'), [history])
  const amtSaved = useMemo(
    () => fraudTxns.reduce((sum, item) => sum + Number(item.amount || 0), 0),
    [fraudTxns],
  )
  const totalTxns = history.length
  const riskPct = totalTxns > 0 ? Math.round((fraudTxns.length / totalTxns) * 100) : 0
  const riskLabel = riskPct < 20 ? 'LOW' : riskPct < 50 ? 'MEDIUM' : 'HIGH'
  const riskColor = riskPct < 20 ? 'var(--green)' : riskPct < 50 ? 'var(--amber)' : 'var(--red)'
  const alertDot = alerts.some((a) => a.type === 'fraud')
  const welcomeName = account.fname ? account.fname : 'Agent'

  useEffect(() => {
    localStorage.setItem('fv_account', JSON.stringify(account))
  }, [account])

  useEffect(() => {
    localStorage.setItem('fv_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    localStorage.setItem('fv_alerts', JSON.stringify(alerts))
  }, [alerts])

  useEffect(() => {
    if (storedAccount) {
      setForm(storedAccount)
    }
  }, [])

  const handleFormChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  const handleTestChange = (field, value) => {
    setTestData((prev) => ({ ...prev, [field]: value }))
  }

  const saveAccount = () => {
    const fname = form.fname.trim()
    if (!fname) {
      window.alert('Please enter your first name!')
      return
    }

    const saved = {
      ...form,
      uid: account.uid || 'ACC-' + Math.floor(Math.random() * 900000 + 100000),
      joined: account.joined || new Date().toLocaleDateString(),
    }
    setAccount(saved)
    setForm(saved)
    setToastVisible(true)
    setTimeout(() => setToastVisible(false), 2500)
    setTab('profile')
  }

  const clearHistory = () => {
    if (window.confirm('Clear all transaction history?')) {
      setHistory([])
      setAlerts([])
      setResult(null)
    }
  }

  const testTransaction = async () => {
    const tx = {
      amount: Number(testData.amount) || 0,
      hour: Number(testData.hour) || 0,
      location: testData.location,
      device: testData.device,
      merchant: testData.merchant,
      note: testData.note,
      date: new Date().toLocaleDateString(),
    }

    setLoading(true)
    let scored = null
    try {
      const apiResult = await apiScoreTransaction({
        user_id: account.uid || 'user_1',
        amount: tx.amount,
        location: tx.location,
        device_id: tx.device,
        time: `${String(tx.hour).padStart(2, '0')}:00`,
        account_id: account.uid ? `acct_${account.uid.slice(-3)}` : 'acct_001',
        merchant_id: `merchant_${tx.merchant.replace(/\W/g, '').toLowerCase()}`,
      })
      scored = {
        score: Number(apiResult.risk_score),
        verdict: apiResult.verdict,
        reasons: apiResult.reasons.map((text) => ({ label: text, score: 0, type: 'risk' })),
      }
    } catch (error) {
      scored = localScore(tx, account.avgspend)
    }

    const entry = { ...tx, ...scored, id: Date.now() }
    const nextHistory = [...history, entry]
    setHistory(nextHistory)
    setResult(entry)
    setResultFlipped(false)

    if (entry.verdict === 'FRAUD') {
      const alert = {
        type: 'fraud',
        title: `Fraud detected — ₹${Number(entry.amount).toLocaleString('en-IN')} at ${entry.merchant}`,
        sub: `${entry.location} · ${entry.hour}:00 · Risk: ${Math.round(entry.score * 100)}%`,
        time: new Date().toLocaleTimeString(),
      }
      setAlerts((prev) => [...prev, alert])
    }

    setLoading(false)
    setTab('monitor')
  }

  const pageClass = (name) => (tab === name ? 'page-section active' : 'page-section')
  const tabClass = (name) => (tab === name ? 'tab active' : 'tab')

  return (
    <div className="page">
      <nav className="glass">
        <div className="logo">
          <div className="logo-mark">VF</div>
          <div>
            <div className="logo-name">VERIFEX</div>
          </div>
        </div>
        <div className="nav-tabs">
          {!account.uid && (
            <button className={tabClass('setup')} onClick={() => setTab('setup')} id="tab-setup">
              Setup
            </button>
          )}
          <button className={tabClass('profile')} onClick={() => setTab('profile')} id="tab-profile">
            My Account <span className="notif" style={{ display: alertDot ? 'inline-block' : 'none' }} />
          </button>
          <button className={tabClass('monitor')} onClick={() => setTab('monitor')} id="tab-monitor">
            Live Monitor
          </button>
        </div>
      </nav>

      <div className="glass hero" style={{ margin: '20px 0 28px', padding: 28 }}>
        <div>
          <div className="hero-title">Protect your account with smarter transaction monitoring</div>
          <div className="hero-sub">Verify payments instantly, surface abnormal activity, and get clear fraud alerts without needing manual review.</div>
          <div className="chip-list">
            <div className="chip">Alerts: {alerts.length}</div>
            <div className="chip">Transactions checked: {totalTxns}</div>
            <div className="chip">Account status: {account.uid ? 'Protected' : 'Not set'}</div>
          </div>
        </div>
      </div>

      <div className={pageClass('setup')} id="page-setup">
        <div className="setup-wrap">
          <div className="glass" style={{ padding: 32 }}>
            <div className="setup-title">Create Your Account</div>
            <div className="setup-sub">Enter your details to start tracking your fraud protection</div>

            <div className="avatar-wrap">
              <div>
                <div className="avatar" id="avatar-display">
                  {formatInitials(form.fname, form.lname)}
                </div>
                <div className="avatar-hint">Your initials</div>
              </div>
            </div>

            <div className="form-grid">
              <div className="fg">
                <label className="fl">First Name</label>
                <input className="fi" type="text" value={form.fname} onChange={(e) => handleFormChange('fname', e.target.value)} />
              </div>
              <div className="fg">
                <label className="fl">Last Name</label>
                <input className="fi" type="text" value={form.lname} onChange={(e) => handleFormChange('lname', e.target.value)} />
              </div>
            </div>

            <div className="form-grid">
              <div className="fg">
                <label className="fl">Email</label>
                <input className="fi" type="email" value={form.email} onChange={(e) => handleFormChange('email', e.target.value)} />
              </div>
              <div className="fg">
                <label className="fl">Phone</label>
                <input className="fi" type="tel" value={form.phone} onChange={(e) => handleFormChange('phone', e.target.value)} />
              </div>
            </div>

            <div className="form-grid">
              <div className="fg">
                <label className="fl">Bank Name</label>
                <select className="fi" value={form.bank} onChange={(e) => handleFormChange('bank', e.target.value)}>
                  <option>SBI</option>
                  <option>HDFC Bank</option>
                  <option>ICICI Bank</option>
                  <option>Axis Bank</option>
                  <option>Kotak Bank</option>
                  <option>Yes Bank</option>
                  <option>Other</option>
                </select>
              </div>
              <div className="fg">
                <label className="fl">Account Type</label>
                <select className="fi" value={form.acctype} onChange={(e) => handleFormChange('acctype', e.target.value)}>
                  <option>Savings</option>
                  <option>Current</option>
                  <option>Salary</option>
                </select>
              </div>
            </div>

            <div className="form-grid">
              <div className="fg">
                <label className="fl">Monthly Avg Spend (₹)</label>
                <input className="fi" type="number" value={form.avgspend} onChange={(e) => handleFormChange('avgspend', e.target.value)} />
              </div>
              <div className="fg">
                <label className="fl">City</label>
                <input className="fi" type="text" value={form.city} onChange={(e) => handleFormChange('city', e.target.value)} />
              </div>
            </div>

            <div className="form-grid full">
              <div className="fg">
                <label className="fl">Alert Threshold</label>
                <select className="fi" value={form.threshold} onChange={(e) => handleFormChange('threshold', e.target.value)}>
                  <option value="low">Low — Alert me on everything suspicious</option>
                  <option value="medium">Medium — Alert on likely fraud only</option>
                  <option value="high">High — Alert only on confirmed fraud</option>
                </select>
              </div>
            </div>

            <button className="save-btn" onClick={saveAccount}>
              ▶ SAVE & START MONITORING
            </button>
          </div>
        </div>
      </div>

      <div className={pageClass('profile')} id="page-profile">
        <div className="stats4">
          <div className="glass scard r">
            <div className="sg" />
            <div className="slabel">Frauds Blocked</div>
            <div className="sval">{fraudTxns.length}</div>
            <div className="ssub">this month</div>
          </div>
          <div className="glass scard g">
            <div className="sg" />
            <div className="slabel">Safe Transactions</div>
            <div className="sval">{safeTxns.length}</div>
            <div className="ssub">cleared</div>
          </div>
          <div className="glass scard a">
            <div className="sg" />
            <div className="slabel">Amount Protected</div>
            <div className="sval">₹{amtSaved.toLocaleString('en-IN')}</div>
            <div className="ssub">saved from fraud</div>
          </div>
          <div className="glass scard b">
            <div className="sg" />
            <div className="slabel">Risk Score</div>
            <div className="sval" style={{ color: riskColor }}>{riskLabel}</div>
            <div className="ssub">account safety</div>
          </div>
        </div>

        <div className="profile-grid">
          <div className="glass profile-card">
            <div className="profile-avatar">{formatInitials(account.fname, account.lname)}</div>
            <div className="profile-name">{account.fname} {account.lname}</div>
            <div className="profile-uid">{account.uid || 'ACC-000000'}</div>
            <div className="profile-badge">✓ Protected Account</div>
            <div className="divider" />
            <div className="info-row"><span className="info-key">Email</span><span className="info-val">{account.email || '—'}</span></div>
            <div className="info-row"><span className="info-key">Phone</span><span className="info-val">{account.phone || '—'}</span></div>
            <div className="info-row"><span className="info-key">Bank</span><span className="info-val">{account.bank}</span></div>
            <div className="info-row"><span className="info-key">Account</span><span className="info-val">{account.acctype}</span></div>
            <div className="info-row"><span className="info-key">City</span><span className="info-val">{account.city || '—'}</span></div>
            <div className="info-row"><span className="info-key">Avg Spend</span><span className="info-val">₹{Number(account.avgspend || 0).toLocaleString('en-IN')}</span></div>
            <div className="divider" />
            <div className="risk-meter-wrap" style={{ padding: 0, width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: 'var(--muted)' }}>Account Risk Level</span>
                <span className="info-val" style={{ color: riskColor }}>{riskLabel}</span>
              </div>
              <div className="risk-bar-bg">
                <div className="risk-bar-fill" style={{ width: Math.max(5, riskPct) + '%', background: riskColor }} />
              </div>
              <div className="risk-labels"><span>Safe</span><span>Medium</span><span>High</span></div>
            </div>
            <button className="edit-btn" onClick={() => setTab('setup')} style={{ width: '100%', marginTop: 8 }}>
              Edit Profile
            </button>
          </div>

          <div>
            <div className="glass" style={{ marginBottom: 16 }}>
              <div className="panel-head">
                <span className="panel-title">Recent Alerts</span>
                <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--mono)' }}>{alerts.length} alerts</span>
              </div>
              <div id="alerts-list" style={{ padding: '12px 16px', maxHeight: 220, overflowY: 'auto' }}>
                {alerts.length === 0 ? (
                  <div className="empty"><div className="empty-icon">🔔</div><span style={{ fontSize: 13 }}>No alerts yet — monitoring active</span></div>
                ) : alerts.slice().reverse().map((a, index) => (
                  <div className={`alert ${a.type}`} key={index}>
                    <div className="alert-icon">{a.type === 'fraud' ? '🚨' : '✅'}</div>
                    <div className="alert-text">
                      <div className="alert-title">{a.title}</div>
                      <div className="alert-sub">{a.sub}</div>
                    </div>
                    <div className="alert-time">{a.time}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass">
              <div className="panel-head">
                <span className="panel-title">Transaction History</span>
                <button className="edit-btn" onClick={clearHistory}>Clear</button>
              </div>
              <div id="txn-history" style={{ maxHeight: 260, overflowY: 'auto' }}>
                {history.length === 0 ? (
                  <div className="empty"><div className="empty-icon">💳</div><span style={{ fontSize: 13 }}>No transactions yet</span></div>
                ) : history.slice().reverse().map((t) => (
                  <div className="txn-item" key={t.id}>
                    <div className={`txn-icon ${t.verdict === 'FRAUD' ? 'fraud' : 'legit'}`}>
                      {t.verdict === 'FRAUD' ? '🚨' : '✅'}
                    </div>
                    <div className="txn-info">
                      <div className="txn-top">
                        <span className="txn-name">{t.merchant}</span>
                        <span className={`txn-amt ${t.verdict === 'FRAUD' ? 'fraud' : 'legit'}`}>₹{Number(t.amount).toLocaleString('en-IN')}</span>
                      </div>
                      <div className="txn-meta">{t.location} · {t.hour}:00 · {t.date}</div>
                    </div>
                    <span className={`badge-sm ${t.verdict === 'FRAUD' ? 'fraud' : 'legit'}`}>{t.verdict}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className={pageClass('monitor')} id="page-monitor">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div className="glass">
            <div className="panel-head"><span className="panel-title">Test A Transaction</span></div>
            <div style={{ padding: '16px 20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div className="fg">
                <label className="fl">Amount (₹)</label>
                <input className="fi" type="number" value={testData.amount} onChange={(e) => handleTestChange('amount', e.target.value)} />
              </div>
              <div className="fg">
                <label className="fl">Hour (0-23)</label>
                <input className="fi" type="number" min="0" max="23" value={testData.hour} onChange={(e) => handleTestChange('hour', e.target.value)} />
              </div>
              <div className="fg">
                <label className="fl">Location</label>
                <select className="fi" value={testData.location} onChange={(e) => handleTestChange('location', e.target.value)}>
                  <option>Mumbai</option>
                  <option>Delhi</option>
                  <option>Bengaluru</option>
                  <option>Unknown</option>
                  <option>Foreign</option>
                </select>
              </div>
              <div className="fg">
                <label className="fl">Device</label>
                <select className="fi" value={testData.device} onChange={(e) => handleTestChange('device', e.target.value)}>
                  <option value="known_device">My Phone (known)</option>
                  <option value="new_device_9999">Unknown Device</option>
                </select>
              </div>
              <div className="fg">
                <label className="fl">Merchant</label>
                <select className="fi" value={testData.merchant} onChange={(e) => handleTestChange('merchant', e.target.value)}>
                  <option>Amazon</option>
                  <option>Swiggy</option>
                  <option>Crypto Exchange</option>
                  <option>Unknown Merchant</option>
                  <option>ATM Withdrawal</option>
                </select>
              </div>
              <div className="fg">
                <label className="fl">Notes</label>
                <input className="fi" type="text" value={testData.note} onChange={(e) => handleTestChange('note', e.target.value)} placeholder="optional" />
              </div>
              <button className="save-btn" style={{ gridColumn: '1 / -1' }} onClick={testTransaction} disabled={loading}>
                {loading ? 'Checking…' : '▶ CHECK THIS TRANSACTION'}
              </button>
            </div>
          </div>

          <div className="glass">
            <div className="panel-head">
              <span className="panel-title">Result</span>
              {result && (
                <button className="flip-toggle" onClick={() => setResultFlipped((prev) => !prev)}>
                  {resultFlipped ? 'Show Summary' : 'Show Details'}
                </button>
              )}
            </div>
            <div id="result-wrap" style={{ padding: '24px 20px' }}>
              {!result ? (
                <div className="empty"><div className="empty-icon">🔍</div><span style={{ fontSize: 13 }}>Enter transaction details and check</span></div>
              ) : (
                <div className="flip-card-outer">
                  <div className={`flip-card ${resultFlipped ? 'flipped' : ''}`}>
                    <div className="flip-card-inner">
                      <div className="flip-card-face flip-card-front">
                        <div style={{ textAlign: 'center', padding: '10px 0 18px' }}>
                          <div className="result-score" style={{ color: result.verdict === 'FRAUD' ? 'var(--red)' : 'var(--green)' }}>
                            {Math.round(result.score * 100)}%
                          </div>
                          <div className="result-verdict" style={{ color: result.verdict === 'FRAUD' ? 'var(--red)' : 'var(--green)' }}>
                            {result.verdict}
                          </div>
                          <div className="result-sub">Risk Score</div>
                        </div>
                        <div className="result-summary">
                          <p>{result.reasons.slice(0, 2).map((item) => item.label).join(' · ')}</p>
                          <div className="card-flags">
                            <span className={`badge-pill ${result.verdict === 'FRAUD' ? 'badge-fraud' : 'badge-legit'}`}>{result.verdict}</span>
                            <span className="badge-pill">{history.length} checks</span>
                          </div>
                        </div>
                      </div>
                      <div className="flip-card-face flip-card-back">
                        <div style={{ padding: '12px 0' }}>
                          <div className="result-label">Why this transaction was scored</div>
                          <div className="back-list">
                            {result.reasons.map((reason, idx) => (
                              <div key={idx} className="back-item">
                                <span>{reason.label}</span>
                                <span>{reason.score.toFixed(2)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                        {result.contributions && Object.keys(result.contributions).length > 0 && (
                          <div style={{ marginTop: 16 }}>
                            <div className="result-label">Contribution breakdown</div>
                            {Object.entries(result.contributions).map(([key, value]) => (
                              <div key={key} className="back-item" style={{ justifyContent: 'space-between' }}>
                                <span>{key.replace(/_/g, ' ')}</span>
                                <span>{Number(value).toFixed(3)}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="glass">
          <div className="panel-head">
            <span className="panel-title">Live Transaction Feed</span>
            <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--mono)' }}>Watch the latest suspicious activity</span>
          </div>
          <div style={{ padding: '18px 20px' }}>
            <TransactionFeed />
          </div>
        </div>

        <div className="glass" style={{ marginTop: 16 }}>
          <div className="panel-head">
            <span className="panel-title">My Transaction Log</span>
            <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--mono)' }}>{history.length} entries</span>
          </div>
          <div id="txn-log" style={{ maxHeight: 300, overflowY: 'auto' }}>
            {history.length === 0 ? (
              <div className="empty"><div className="empty-icon">📋</div><span style={{ fontSize: 13 }}>Your checked transactions will appear here</span></div>
            ) : history.slice().reverse().map((t) => (
              <div className="txn-item" key={t.id}>
                <div className={`txn-icon ${t.verdict === 'FRAUD' ? 'fraud' : 'legit'}`}>
                  {t.verdict === 'FRAUD' ? '🚨' : '✅'}
                </div>
                <div className="txn-info">
                  <div className="txn-top">
                    <span className="txn-name">{t.merchant} · {t.location}</span>
                    <span className={`txn-amt ${t.verdict === 'FRAUD' ? 'fraud' : 'legit'}`}>₹{Number(t.amount).toLocaleString('en-IN')}</span>
                  </div>
                  <div className="txn-meta">{t.note || 'No notes'} · {t.date} · Risk: {Math.round(t.score * 100)}%</div>
                </div>
                <span className={`badge-sm ${t.verdict === 'FRAUD' ? 'fraud' : 'legit'}`}>{t.verdict}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="toast" style={{ display: toastVisible ? 'block' : 'none' }}>✓ Account saved!</div>
    </div>
  )
}
