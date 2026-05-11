import { useEffect, useState } from 'react'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function AdminDashboardPage({ user, onLogout }) {
  const [overview, setOverview] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    async function loadOverview() {
      try {
        const data = await api.get('/admin/overview')
        setOverview(data)
      } catch (err) {
        setError(err.message)
      }
    }
    loadOverview()
  }, [])

  return (
    <Layout user={user} onLogout={onLogout} title="Admin Dashboard" subtitle="Review system users and overall usage." admin>
      <div className="dashboard-grid">
        <section className="stats-grid">
          <article className="stat-card"><span className="stat-label">Total Users</span><strong>{overview?.totals?.users ?? 0}</strong></article>
          <article className="stat-card"><span className="stat-label">Essay Reports</span><strong>{overview?.totals?.essays ?? 0}</strong></article>
          <article className="stat-card"><span className="stat-label">Question Sets</span><strong>{overview?.totals?.questionSets ?? 0}</strong></article>
        </section>
        <section className="panel">
          <h3>Registered Users</h3>
          {overview?.users?.length ? overview.users.map((item) => (
            <div className="row-item resource-row" key={item.id}>
              <div>
                <strong>{item.full_name}</strong>
                <p>{item.email} • {item.role}</p>
              </div>
              <small>{new Date(item.created_at).toLocaleString()}</small>
            </div>
          )) : <p className="muted-text">No users found.</p>}
        </section>
      </div>
      {error && <div className="floating-message error">{error}</div>}
    </Layout>
  )
}
