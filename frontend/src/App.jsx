import React from 'react'
import VerificationForm from './components/VerificationForm'
import TransactionFeed from './components/TransactionFeed'

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Verifex Fraud Detection</p>
          <h1>Transaction Scoring Dashboard</h1>
          <p className="subtitle">
            Verify transactions instantly, inspect model explanations, and track suspicious activity from one dashboard.
          </p>
        </div>
      </header>
      <main className="grid-two-column">
        <section className="card">
          <VerificationForm />
        </section>
        <section className="card">
          <TransactionFeed />
        </section>
      </main>
    </div>
  )
}
