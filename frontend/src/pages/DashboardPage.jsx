import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function DashboardPage({ user, onLogout }) {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')
  const [selectedEssay, setSelectedEssay] = useState(null)
  const [selectedQuestionSet, setSelectedQuestionSet] = useState(null)

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

async function openEssayReport(essay) {
  try {
    const fullReport = await api.get(`/reports/${essay.id}`)
    setSelectedEssay(fullReport.report || fullReport)
  } catch (error) {
    window.alert('Could not load full report.')
  }
}
  return (
    <Layout user={user} onLogout={onLogout} title="Lecturer Dashboard" subtitle="Manage the complete workflow from evaluation to stored resources.">
      <div className="dashboard-grid">
        <section className="panel hero-panel refined-hero">
  <div className="hero-top">
    <div>
      <p className="eyebrow">WELCOME BACK</p>

      <h1 className="dashboard-name">
        {user.fullName}
      </h1>

      <p className="hero-description">
          Continue evaluating student essays, generating comprehension questions,
          and reviewing saved academic work from one central dashboard.
      </p>
    </div>

    <div className="hero-badge-card">
      <span className="hero-badge-number">
        {summary?.totals?.essays ?? 0}
      </span>

      <span className="hero-badge-label">
        Essays Evaluated
      </span>
    </div>
  </div>

  <div className="dashboard-quick-actions">
    <Link className="primary-btn hero-main-btn" to="/essay-evaluation">
        Start New Evaluation
    </Link>

    <Link className="secondary-btn" to="/question-generation">
        Create Question Set
    </Link>

    <Link className="secondary-btn" to="/resources">
        View Saved Work
    </Link>
  </div>
          </section>

        <section className="stats-grid refined-stats">
  <article className="stat-card compact-stat">
    <div>
      <span className="stat-label">Total Essays</span>
      <strong>{summary?.totals?.essays ?? 0}</strong>
    </div>

    <div className="stat-icon">📝</div>
  </article>

  <article className="stat-card compact-stat">
    <div>
      <span className="stat-label">Sentence Checks</span>
      <strong>{summary?.totals?.sentences ?? 0}</strong>
    </div>

    <div className="stat-icon">📊</div>
  </article>

  <article className="stat-card compact-stat">
    <div>
      <span className="stat-label">Question Sets</span>
      <strong>{summary?.totals?.questionSets ?? 0}</strong>
    </div>

    <div className="stat-icon">❓</div>
  </article>

  <article className="stat-card compact-stat">
    <div>
      <span className="stat-label">Top Issue</span>
      <strong>{summary?.totals?.topCategory ?? 'None'}</strong>
    </div>

    <div className="stat-icon">⚠️</div>
  </article>
</section>

<section className="panel dashboard-snapshot">
  <div>
    <p className="eyebrow">ACTIVITY SNAPSHOT</p>
    <h3>Evaluation progress</h3>
    <p>
      Your most frequent issue category is
      <strong> {summary?.totals?.topCategory ?? 'No data yet'}</strong>.
    </p>
  </div>

  <div className="snapshot-bars">
    <div><span>Essays</span><b style={{ width: '72%' }}></b></div>
    <div><span>Questions</span><b style={{ width: '55%' }}></b></div>
    <div><span>Reports</span><b style={{ width: '64%' }}></b></div>
  </div>
</section>

        <section className="panel">
          <h3>Recent Essay Reports</h3>
          {summary?.recentEssays?.length ? (
            <div className="list-grid">
              {summary.recentEssays.map((essay) => (
  <div
    className="dashboard-record-card clickable-card"
    key={essay.id}
    onClick={() => openEssayReport(essay)}
  >
    <div className="record-main">
      <strong>{essay.title}</strong>

      <p>
        {essay.sentenceCount} sentences • {essay.rating}
      </p>
    </div>

    <span className="record-date">
      {new Date(essay.createdAt).toLocaleDateString()}
    </span>
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
                <div
                    className="row-item clickable-card"
                    key={item.id}
                    onClick={() => setSelectedQuestionSet(item)}
                >
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
          
        {selectedEssay && (
  <div
    className="modal-overlay"
    onClick={() => setSelectedEssay(null)}
  >
    <div
      className="report-modal"
      onClick={(e) => e.stopPropagation()}
    >
      <button
        className="modal-close"
        onClick={() => setSelectedEssay(null)}
      >
        ✕
      </button>

      <h2>{selectedEssay.title}</h2>

      <p>
        <p>Sentence Count: {selectedEssay.sentenceCount || selectedEssay.sentence_count || 'N/A'}</p>
      </p>

      <p>
        Sentence Count: {selectedEssay.sentenceCount}
      </p>

      <div className="modal-section">
  <h4>Essay Feedback</h4>
          
  <div className="report-grid">
  <div className="report-item">
    <span>Grammar</span>
    <strong>{selectedEssay.summary?.counts?.Grammar ?? 0} issues</strong>
  </div>

  <div className="report-item">
    <span>Spelling</span>
    <strong>{selectedEssay.summary?.counts?.Spelling ?? 0} issues</strong>
  </div>

  <div className="report-item">
    <span>Punctuation</span>
    <strong>{selectedEssay.summary?.counts?.Punctuation ?? 0} issues</strong>
  </div>

  <div className="report-item">
    <span>Word Usage</span>
    <strong>{selectedEssay.summary?.counts?.Word_Usage ?? 0} issues</strong>
  </div>
</div>

<div className="feedback-box">
  <h4>AI Summary</h4>
  <p>
    Total issues detected: {selectedEssay.essay_metrics?.totalIssues ?? 0}
  </p>
  <p>
    Readability level: {selectedEssay.essay_metrics?.readabilityLevel ?? 'N/A'}
  </p>
  <p>
    Lexical diversity: {selectedEssay.essay_metrics?.lexicalDiversity ?? 'N/A'}
  </p>
</div>

  <div className="report-grid">

    <div className="report-item">
      <span>Grammar</span>
      <strong>
        {selectedEssay.grammarErrors ?? 0} issues
      </strong>
    </div>

    <div className="report-item">
      <span>Spelling</span>
      <strong>
        {selectedEssay.spellingErrors ?? 0} issues
      </strong>
    </div>

    <div className="report-item">
      <span>Punctuation</span>
      <strong>
        {selectedEssay.punctuationErrors ?? 0} issues
      </strong>
    </div>

    <div className="report-item">
      <span>Sentence Structure</span>
      <strong>
        {selectedEssay.structureErrors ?? 0} issues
      </strong>
    </div>

  </div>

  <div className="feedback-box">
    <h4>AI Summary</h4>

    <p>
      {selectedEssay.feedback ||
        'No detailed feedback available.'}
    </p>
  </div>
</div>
    </div>
  </div>
)}
    </Layout>
  )
}
