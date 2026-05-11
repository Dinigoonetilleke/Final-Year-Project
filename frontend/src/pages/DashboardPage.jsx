import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function DashboardPage({ user, onLogout }) {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    async function loadSummary() {
      try {
        const data = await api.get(`/dashboard/summary?userId=${user.id}`)
        setSummary(data)
      } catch (err) {
        setError(err.message)
      }
    }
    loadSummary()
  }, [user.id])

  return (
    <Layout user={user} onLogout={onLogout} title="Lecturer Dashboard" subtitle="Manage the complete workflow from evaluation to stored resources.">
      <div className="dashboard-grid">
        <section className="panel hero-panel">
          <p className="eyebrow">Welcome back</p>
          <h1>{user.fullName}</h1>
          <p className="muted-text">This dashboard gives separated access to essay evaluation, question generation, and the stored reports page.</p>
          <div className="action-row">
            <Link className="primary-btn" to="/essay-evaluation">Evaluate an Essay</Link>
            <Link className="secondary-btn" to="/question-generation">Generate Questions</Link>
            <Link className="secondary-btn" to="/resources">Open Storage</Link>
            <Link className="secondary-btn" to="/reports"> View Essay Reports </Link>
          </div>
        </section>

        <section className="stats-grid">
          <article className="stat-card"><span className="stat-label">Total Essays</span><strong>{summary?.totals?.essays ?? 0}</strong></article>
          <article className="stat-card"><span className="stat-label">Sentence Checks</span><strong>{summary?.totals?.sentences ?? 0}</strong></article>
          <article className="stat-card"><span className="stat-label">Question Sets</span><strong>{summary?.totals?.questionSets ?? 0}</strong></article>
          <article className="stat-card"><span className="stat-label">Top Issue Category</span><strong>{summary?.totals?.topCategory ?? 'No data yet'}</strong></article>
        </section>

        <section className="panel">
          <h3>Recent Essay Reports</h3>
          {summary?.recentEssays?.length ? (
            <div className="list-grid">
              {summary.recentEssays.map((essay) => (
                <div className="row-item" key={essay.id}>
                  <strong>{essay.title}</strong>
                  <p>{essay.sentenceCount} sentences • {essay.rating}</p>
                  <small>{new Date(essay.createdAt).toLocaleString()}</small>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted-text">No essay reports saved yet.</p>
          )}
        </section>

        <section className="panel">
          <h3>Recent Question Sets</h3>
          {summary?.recentQuestions?.length ? (
            <div className="list-grid">
              {summary.recentQuestions.map((item) => (
                <div className="row-item" key={item.id}>
                  <strong>{item.title}</strong>
                  <p>{item.questionCount} questions generated</p>
                  <small>{new Date(item.createdAt).toLocaleString()}</small>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted-text">No question sets saved yet.</p>
          )}
        </section>
      </div>
      {error && <div className="floating-message error">{error}</div>}
    </Layout>
  )
}
