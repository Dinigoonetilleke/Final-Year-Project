function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default function Dashboard({ user, summary, onNavigate, onLogout }) {
  return (
    <div className="page-shell">
      <header className="top-nav">
        <div>
          <h2>Essay Evaluation System</h2>
          <p>Lecturer Dashboard</p>
        </div>
        <div className="top-nav-actions">
          <span className="user-pill">{user.fullName}</span>
          <button className="secondary-btn" onClick={() => onNavigate('evaluate')}>
            Evaluate Essay
          </button>
          <button className="ghost-btn" onClick={onLogout}>
            Logout
          </button>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel hero-panel">
          <h1>Welcome back, Lecturer</h1>
          <p>
            Review submissions, track error categories, and generate structured feedback in a
            clean lecturer-oriented workflow.
          </p>
          <button className="primary-btn" onClick={() => onNavigate('evaluate')}>
            Start New Evaluation
          </button>
        </section>

        <section className="stats-grid">
          <StatCard label="Essays Evaluated" value={summary?.totals?.essays ?? 0} />
          <StatCard label="Sentences Processed" value={summary?.totals?.sentences ?? 0} />
          <StatCard label="Most Frequent Issue" value={summary?.totals?.topCategory ?? 'No data yet'} />
        </section>

        <section className="panel">
          <h3>Error Frequency Summary</h3>
          <div className="list-stack">
            {Object.entries(summary?.categoryTotals ?? {}).length > 0 ? (
              Object.entries(summary.categoryTotals).map(([key, value]) => (
                <div key={key} className="row-item">
                  <span>{key.replaceAll('_', ' ')}</span>
                  <strong>{value}</strong>
                </div>
              ))
            ) : (
              <p className="muted-text">No analytics available yet.</p>
            )}
          </div>
        </section>

        <section className="panel">
          <h3>Recent Reports</h3>
          <div className="list-stack">
            {summary?.recentEssays?.length ? (
              summary.recentEssays.map((essay) => (
                <div key={essay.id} className="row-item wrap-row">
                  <div>
                    <strong>{essay.title}</strong>
                    <p>{new Date(essay.createdAt).toLocaleString()}</p>
                  </div>
                  <span>{essay.sentenceCount} sentences</span>
                </div>
              ))
            ) : (
              <p className="muted-text">No reports saved yet.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  )
}
