import { useEffect, useMemo, useState } from 'react'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function AdminDashboardPage({ user, onLogout }) {
  const [overview, setOverview] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard')
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

  const lecturers = users.filter(
    (u) => u.role === 'lecturer'
  )

  const admins = users.filter(
    (u) => u.role === 'admin'
  )

  const filteredUsers = useMemo(() => {
    return users.filter((item) => {
      const text =
        `${item.full_name || ''} ${item.email || ''} ${item.role || ''}`.toLowerCase()

      return text.includes(search.toLowerCase())
    })
  }, [users, search])

  return (
    <Layout
      user={user}
      onLogout={onLogout}
      title="Admin Dashboard"
      subtitle="System-wide overview of lecturers, essay records, AI evaluations, and generated questions."
      admin={true}
    >
      <div className="admin-page">

        <div className="admin-topbar">
          <div>
            <h2>Admin Overview</h2>
            <p>
              Monitor users, essay reports,
              question sets, and system activity.
            </p>
          </div>

          <input
            className="admin-search"
            type="text"
            placeholder="Search users..."
            value={search}
            onChange={(e) =>
              setSearch(e.target.value)
            }
          />
        </div>

        <div className="admin-tabs">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={
              activeTab === 'dashboard'
                ? 'active'
                : ''
            }
          >
            Dashboard
          </button>

          <button
            onClick={() => setActiveTab('lecturers')}
            className={
              activeTab === 'lecturers'
                ? 'active'
                : ''
            }
          >
            Lecturers
          </button>

          <button
            onClick={() => setActiveTab('records')}
            className={
              activeTab === 'records'
                ? 'active'
                : ''
            }
          >
            Essay Records
          </button>

          <button
            onClick={() => setActiveTab('system')}
            className={
              activeTab === 'system'
                ? 'active'
                : ''
            }
          >
            System
          </button>
        </div>

        {activeTab === 'dashboard' && (
          <>
            <section className="admin-stats-grid">
              <StatCard
                label="Lecturers"
                value={lecturers.length}
                icon="👩‍🏫"
              />

              <StatCard
                label="Admins"
                value={admins.length}
                icon="🛡️"
              />

              <StatCard
                label="Essay Reports"
                value={overview?.totals?.essays ?? 0}
                icon="📝"
              />

              <StatCard
                label="Question Sets"
                value={overview?.totals?.questionSets ?? 0}
                icon="❔"
              />
            </section>

            <section className="admin-mini-stats">
              <MiniStat
                label="Total Users"
                value={overview?.totals?.users ?? 0}
              />

              <MiniStat
                label="Average Reports Per Lecturer"
                value={
                  lecturers.length
                    ? (
                        (overview?.totals?.essays ?? 0) /
                        lecturers.length
                      ).toFixed(1)
                    : '0.0'
                }
              />

              <MiniStat
                label="System Status"
                value="Active"
              />
            </section>

            <div className="admin-content-grid">
              <section className="admin-panel large">
                <div className="panel-heading">
                  <h3>AI Evaluation Insights</h3>
                  <span>Project analytics</span>
                </div>

                <div className="insight-grid">
                  <SummaryItem
                    label="Most used module"
                    value="Essay Evaluation"
                  />

                  <SummaryItem
                    label="Common error type"
                    value="Grammar"
                  />

                  <SummaryItem
                    label="Primary user role"
                    value="Lecturer"
                  />

                  <SummaryItem
                    label="Feedback mode"
                    value="Editable by lecturer"
                  />
                </div>
              </section>

              <section className="admin-panel">
                <div className="panel-heading">
                  <h3>System Health</h3>
                  <span>Live status</span>
                </div>

                <div className="summary-list">
                  <SummaryItem
                    label="Firebase"
                    value="Connected"
                  />

                  <SummaryItem
                    label="Flask API"
                    value="Active"
                  />

                  <SummaryItem
                    label="AI Model"
                    value="Available"
                  />

                  <SummaryItem
                    label="Role Access"
                    value="Enabled"
                  />
                </div>
              </section>
            </div>
          </>
        )}

        {activeTab === 'lecturers' && (
          <section className="admin-panel">
            <div className="panel-heading">
              <h3>Registered Users</h3>
              <span>
                {filteredUsers.length} found
              </span>
            </div>

            <div className="admin-table">

              <div className="admin-table-head">
                <span>Name</span>
                <span>Email</span>
                <span>Role</span>
                <span>Created</span>
              </div>

              {filteredUsers.length ? (
                filteredUsers.map((item) => (
                  <div
                    className="admin-table-row"
                    key={item.id}
                  >
                    <span>
                      {item.full_name ||
                        'Unnamed User'}
                    </span>

                    <span>{item.email}</span>

                    <span>
                      <b
                        className={
                          'role-pill ' + item.role
                        }
                      >
                        {item.role}
                      </b>
                    </span>

                    <span>
                      {item.created_at
                        ? new Date(
                            item.created_at
                          ).toLocaleDateString()
                        : 'N/A'}
                    </span>
                  </div>
                ))
              ) : (
                <p className="muted-text">
                  No users found.
                </p>
              )}
            </div>
          </section>
        )}

        {activeTab === 'records' && (
          <section className="admin-panel">
            <div className="panel-heading">
              <h3>Essay Records</h3>
              <span>
                Saved evaluation reports
              </span>
            </div>

            <div className="empty-state">
              <h4>Essay records summary</h4>

              <p>
                Total saved essay reports:
                {' '}
                {overview?.totals?.essays ?? 0}
              </p>
            </div>
          </section>
        )}

        {activeTab === 'system' && (
          <section className="admin-panel">
            <div className="panel-heading">
              <h3>System Monitoring</h3>
              <span>Admin only</span>
            </div>

            <div className="system-grid">
              <SummaryItem
                label="Frontend"
                value="React Active"
              />

              <SummaryItem
                label="Backend"
                value="Flask Active"
              />

              <SummaryItem
                label="Database"
                value="Firebase"
              />

              <SummaryItem
                label="Authentication"
                value="Role-based"
              />
            </div>
          </section>
        )}

        {error && (
          <div className="floating-message error">
            {error}
          </div>
        )}
      </div>
    </Layout>
  )
}

function StatCard({ label, value, icon }) {
  return (
    <article className="admin-stat-card">
      <div className="stat-icon">
        {icon}
      </div>

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