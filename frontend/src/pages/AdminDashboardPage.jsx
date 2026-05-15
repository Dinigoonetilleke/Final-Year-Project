import { useEffect, useMemo, useState } from 'react'
import Layout from '../components/Layout'
import { api } from '../lib/api'

import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

export default function AdminDashboardPage({ user, onLogout }) {
  const [overview, setOverview] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [search, setSearch] = useState('')
  const [error, setError] = useState('')
  const [reports, setReports] = useState([])

  useEffect(() => {
    async function loadOverview() {
      try {
        const data = await api.get('/admin/overview')
        setOverview(data)
          
        const reportData = await api.get('/reports')
        setReports(reportData.reports || [])
          
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

  /* =========================
     CHART DATA
  ========================= */

  const evaluationTrend = [
    { day: 'Mon', essays: 4 },
    { day: 'Tue', essays: 7 },
    { day: 'Wed', essays: 5 },
    { day: 'Thu', essays: 10 },
    { day: 'Fri', essays: 8 },
    { day: 'Sat', essays: 3 },
    { day: 'Sun', essays: 6 },
  ]

  const errorBreakdown = [
    { type: 'Grammar', count: 18 },
    { type: 'Spelling', count: 9 },
    { type: 'Punctuation', count: 12 },
    { type: 'Sentence', count: 7 },
  ]

  const scoreDistribution = [
    { range: '0-40', count: 5 },
    { range: '41-55', count: 8 },
    { range: '56-70', count: 16 },
    { range: '71-85', count: 10 },
    { range: '86-100', count: 5 },
  ]

async function handleDeleteUser(userId) {
  const confirmDelete = window.confirm(
    'Delete this user?'
  )

  if (!confirmDelete) return

  try {
    await api.delete(`/admin/users/${userId}`)

    setOverview((prev) => ({
      ...prev,
      users: prev.users.filter(
        (u) => u.id !== userId
      )
    }))
  } catch (error) {
    alert('Failed to delete user.')
  }
}

async function handleRoleChange(userId, role) {
  try {
    await api.put(
      `/admin/users/${userId}/role`,
      { role }
    )

    setOverview((prev) => ({
      ...prev,
      users: prev.users.map((u) =>
        u.id === userId
          ? { ...u, role }
          : u
      )
    }))

    window.alert(`Role changed successfully to ${role}.`)
  } catch (error) {
    window.alert('Failed to update role.')
  }
}

  return (
    <Layout
        user={user}
        onLogout={onLogout}
        title="Admin Dashboard"
        subtitle="System-wide overview of lecturers, essay records, AI evaluations, and generated questions."
        admin={true}
        activeAdminTab={activeTab}
        onAdminTabChange={setActiveTab}
    >
      <div className="admin-page">

        {/* =========================
            TOP BAR
        ========================= */}

        <div className="admin-topbar">
          <div>
            <h2>Admin Overview</h2>

            <p>
              Monitor users, essay reports,
              question sets, and system activity.
            </p>
          </div>

          {activeTab === 'lecturers' && (
            <input
                className="admin-search"
                type="text"
                placeholder="Search users..."
                value={search}
                onChange={(e) =>
                    setSearch(e.target.value)
                }
            />
        )}
        </div>


        {/* =========================
            DASHBOARD TAB
        ========================= */}

        {activeTab === 'dashboard' && (
          <>

            {/* STAT CARDS */}

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

            {/* MINI STATS */}

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

            {/* CHARTS */}

            <div className="admin-chart-grid">

              {/* LINE CHART */}

              <section className="admin-panel chart-panel">

                <div className="panel-heading">
                  <h3>Evaluations This Week</h3>
                  <span>Essay activity</span>
                </div>

                <ResponsiveContainer width="100%" height={260}>

                  <LineChart data={evaluationTrend}>

                    <XAxis dataKey="day" />

                    <YAxis />

                    <Tooltip />

                    <Line
                      type="monotone"
                      dataKey="essays"
                      stroke="#2563eb"
                      strokeWidth={3}
                      dot={{ r: 5 }}
                    />

                  </LineChart>

                </ResponsiveContainer>

              </section>

              {/* ERROR CHART */}

              <section className="admin-panel chart-panel">

                <div className="panel-heading">
                  <h3>Error Type Breakdown</h3>
                  <span>AI analysis</span>
                </div>

                <ResponsiveContainer width="100%" height={260}>

                  <BarChart data={errorBreakdown}>

                    <XAxis dataKey="type" />

                    <YAxis />

                    <Tooltip />

                    <Bar
                      dataKey="count"
                      fill="#0f172a"
                      radius={[8, 8, 0, 0]}
                    />

                  </BarChart>

                </ResponsiveContainer>

              </section>

              {/* SCORE CHART */}

              <section className="admin-panel chart-panel">

                <div className="panel-heading">
                  <h3>Score Distribution</h3>
                  <span>Essay scores</span>
                </div>

                <ResponsiveContainer width="100%" height={260}>

                  <BarChart data={scoreDistribution}>

                    <XAxis dataKey="range" />

                    <YAxis />

                    <Tooltip />

                    <Bar
                      dataKey="count"
                      fill="#06b6d4"
                      radius={[8, 8, 0, 0]}
                    />

                  </BarChart>

                </ResponsiveContainer>

              </section>

              {/* SYSTEM HEALTH */}

              <section className="admin-panel chart-panel">

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

        {/* =========================
            LECTURERS TAB
        ========================= */}

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
                <span>Actions</span>
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
                    
                    <div className="action-buttons">

                        <button
                            className="edit-btn"
                            onClick={() =>
                                handleRoleChange(
                                    item.id,
                                    item.role === 'admin'
                                        ? 'lecturer'
                                        : 'admin'
                                    )
                            }
                        >
                            Change Role
                        </button>

                        <button
                            className="delete-btn"
                            onClick={() => handleDeleteUser(item.id)}
                        >
                            Delete
                        </button>

                    </div>

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

        {/* =========================
            RECORDS TAB
        ========================= */}

        {activeTab === 'records' && (
  <>
    <section className="admin-mini-stats">
      <MiniStat
        label="Total Essay Reports"
        value={overview?.totals?.essays ?? 0}
      />

      <MiniStat
        label="Average Score"
        value="67%"
      />

      <MiniStat
        label="Most Common Error"
        value="Grammar"
      />
    </section>

    <section className="admin-panel">
      <div className="panel-heading">
        <h3>Recent Essay Records</h3>
        <span>Saved evaluation reports</span>
      </div>

{reports.length ? (
  reports.slice(0, 8).map((report) => (
    <div
      className="admin-table-row records-row"
      key={report.id}
    >
      <span>
        {report.title || 'Untitled Essay'}
      </span>

      <span>
        {report.studentName || 'Unknown'}
      </span>

      <span>
        <b className="role-pill lecturer">
          {report.rating || 'Reviewed'}
        </b>
      </span>

      <span>
        {report.wordCount || 0}
      </span>

      <span>
        {report.created_at
          ? new Date(
              report.created_at
            ).toLocaleDateString()
          : 'N/A'}
      </span>
    </div>
  ))
) : (
  <p className="muted-text">
    No reports found.
  </p>
)}
    </section>
  </>
)}
        {/* =========================
            SYSTEM TAB
        ========================= */}

{activeTab === 'system' && (
  <>
    <section className="admin-stats-grid">
      <StatCard label="API Health" value="Online" icon="🟢" />
      <StatCard label="AI Model" value="Active" icon="🤖" />
      <StatCard label="Firebase" value="Connected" icon="🔥" />
      <StatCard label="OCR" value="Ready" icon="📷" />
    </section>

    <section className="admin-panel">
      <div className="panel-heading">
        <h3>System Monitoring</h3>
        <span>Admin only</span>
      </div>

      <div className="system-grid">
        <SummaryItem label="Frontend" value="React Active" />
        <SummaryItem label="Backend API" value="Flask Running" />
        <SummaryItem label="Database" value="Firebase Firestore" />
        <SummaryItem label="Authentication" value="Role-based Access" />
        <SummaryItem label="AI Evaluation Model" value="Available" />
        <SummaryItem label="OCR Upload Support" value="Image/PDF Enabled" />
        <SummaryItem label="Question Generator" value="Available" />
        <SummaryItem label="Last Updated" value={new Date().toLocaleString()} />
      </div>
    </section>
  </>
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

      <div className="stat-content">
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