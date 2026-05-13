import { useEffect, useMemo, useState } from 'react'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function AdminDashboardPage({ user, onLogout }) {
  const [overview, setOverview] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [search, setSearch] = useState('')
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

  const users = overview?.users ?? []

  const filteredUsers = useMemo(() => {
    return users.filter((item) => {
      const text = `${item.full_name || ''} ${item.email || ''} ${item.role || ''}.toLowerCase()
      return text.includes(search.toLowerCase())
    })
  }, [users, search])

  const lecturers = users.filter((u) => u.role === 'lecturer')
  const admins = users.filter((u) => u.role === 'admin')

  return (
    <Layout
      user={user}
      onLogout={onLogout}
      title="Admin Dashboard"
      subtitle="System-wide overview of lecturers, essay records, AI evaluations, and generated questions."
      admin
    >
      <div className="admin-page">
        <div className="admin-topbar">
          <div>
            <h2>Admin Overview</h2>
            <p>Monitor platform activity and manage academic users.</p>
          </div>

          <input
            className="admin-search"
            type="text"
            placeholder="Search users..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <section className="admin-stats-grid">
          <StatCard label="Lecturers" value={lecturers.length} icon="👩‍🏫" />
          <StatCard label="Admins" value={admins.length} icon="🛡️" />
          <StatCard label="Essay Reports" value={overview?.totals?.essays ?? 0} icon="📝" />
          <StatCard label="Question Sets" value={overview?.totals?.questionSets ?? 0} icon="❔" />
        </section>

        <section className="admin-mini-stats">
          <MiniStat label="Total Users" value={overview?.totals?.users ?? 0} />
          <MiniStat label="Average Reports per Lecturer" value={lecturers.length ? ((overview?.totals?.essays ?? 0) / lecturers.length).toFixed(1) : '0.0'} />
          <MiniStat label="System Status" value="Active" />
        </section>

        <div className="admin-tabs">
          <button className={activeTab === 'overview' ? 'active' : ''} onClick={() => setActiveTab('overview')}>Overview</button>
          <button className={activeTab === 'lecturers' ? 'active' : ''} onClick={() => setActiveTab('lecturers')}>Lecturers</button>
          <button className={activeTab === 'records' ? 'active' : ''} onClick={() => setActiveTab('records')}>Essay Records</button>
          <button className={activeTab === 'settings' ? 'active' : ''} onClick={() => setActiveTab('settings')}>System</button>
        </div>

        {activeTab === 'overview' && (
          <div className="admin-content-grid">
            <section className="admin-panel large">
              <div className="panel-heading">
                <h3>Platform Activity</h3>
                <span>Last 30 days</span>
              </div>
              <div className="empty-chart">
                <div className="chart-line"></div>
                <p>Evaluation activity chart can be connected here later.</p>
              </div>
            </section>

            <section className="admin-panel">
              <div className="panel-heading">
                <h3>Quick Summary</h3>
              </div>

              <div className="summary-list">
                <SummaryItem label="Most used module" value="Essay Evaluation" />
                <SummaryItem label="Primary user role" value="Lecturer" />
                <SummaryItem label="Database" value="Firebase Connected" />
                <SummaryItem label="AI model" value="Available" />
              </div>
            </section>
          </div>
        )}

        {activeTab === 'lecturers' && (
          <section className="admin-panel">
            <div className="panel-heading">
              <h3>Registered Users</h3>
              <span>{filteredUsers.length} found</span>
            </div>

            <div className="admin-table">
              <div className="admin-table-head">
                <span>Name</span>
                <span>Email</span>
                <span>Role</span>
                <span>Created</span>
              </div>

              {filteredUsers.length ? filteredUsers.map((item) => (
                <div className="admin-table-row" key={item.id}>
                  <span>{item.full_name || 'Unnamed User'}</span>
                  <span>{item.email}</span>
                  <span><b className={`role-pill ${item.role}`}>{item.role}</b></span>
                  <span>{item.created_at ? new Date(item.created_at).toLocaleDateString() : 'N/A'}</span>
                </div>
              )) : (
                <p className="muted-text">No users found.</p>
              )}
            </div>
          </section>
        )}

        {activeTab === 'records' && (
          <section className="admin-panel">
            <div className="panel-heading">
              <h3>Essay Records</h3>
              <span>Saved evaluation reports</span>
            </div>

            <div className="empty-state">
              <h4>No detailed essay table connected yet</h4>
              <p>Later, this can show essay title, lecturer, score, error count, and created date.</p>
            </div>
          </section>
        )}

        {activeTab === 'settings' && (
          <section className="admin-panel">
            <div className="panel-heading">
              <h3>System Monitoring</h3>
              <span>Admin only</span>
            </div>

            <div className="system-grid">
              <SummaryItem label="Frontend" value="React Active" />
              <SummaryItem label="Backend" value="Flask Active" />
              <SummaryItem label="Database" value="Firebase" />
              <SummaryItem label="Authentication" value="Role-based" />
            </div>
          </section>
        )}

        {error && <div className="floating-message error">{error}</div>}
      </div>
    </Layout>
  )
}

function StatCard({ label, value, icon }) {
  return (
    <article className="admin-stat-card">
      <div className="stat-icon">{icon}</div>
      <div>
        <strong>{value}</strong>
        <span>{label}</span>
      </div>
    </article>
  )
}

function MiniStat({ label, value }) {
  return (
    <article className="admin-mini-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  )
}

function SummaryItem({ label, value }) {
  return (
    <div className="summary-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}