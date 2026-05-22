import React from 'react'

function Bar({label, value}){
  const pct = Math.min(1, Math.max(0, value))
  const width = Math.round(pct * 100)
  const color = value > 0 ? '#dc2626' : '#16a34a'
  return (
    <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:10}}>
      <div style={{flex:'0 0 150px', fontSize:13, color:'#334155'}}>{label}</div>
      <div style={{flex:1, background:'#f8fafc', height:14, borderRadius:999, overflow:'hidden'}}>
        <div style={{width:`${width}%`,height:'100%',background:color}} />
      </div>
      <div style={{width:56,textAlign:'right',fontSize:13,color:'#334155'}}>{value.toFixed(3)}</div>
    </div>
  )
}

export default function ExplanationPanel({result}){
  if(!result) return null
  const {risk_score, verdict, reasons, contributions} = result
  const entries = Object.entries(contributions || {})
  const statusClass = verdict === 'FRAUD' ? 'badge-fraud' : verdict === 'REVIEW' ? 'badge-review' : 'badge-legit'

  return (
    <div className="result-card">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',gap:18}}>
        <div>
          <p className="result-label">Verdict</p>
          <h2 style={{margin:'0 0 6px',fontSize:'1.2rem'}}>{verdict}</h2>
          <div style={{fontSize:13,color:'#475569'}}>Risk score: <strong>{risk_score.toFixed(3)}</strong></div>
        </div>
        <span className={`badge-pill ${statusClass}`} style={{fontSize:12}}>{verdict}</span>
      </div>

      <div style={{marginTop:18}}>
        <p className="result-label">Why this score?</p>
        <ul style={{paddingLeft:18,margin:'8px 0',color:'#475569',lineHeight:1.7}}>
          {reasons.map((reason, idx) => (
            <li key={idx}>{reason}</li>
          ))}
        </ul>
      </div>

      <div style={{marginTop:18}}>
        {entries.length === 0 ? (
          <div style={{color:'#94a3b8'}}>No specific contributions detected.</div>
        ) : (
          entries.map(([key, value]) => (
            <Bar key={key} label={key.replace(/_/g, ' ')} value={Number(value)} />
          ))
        )}
      </div>
    </div>
  )
}
