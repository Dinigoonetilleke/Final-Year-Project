import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function DashboardPage({ user, onLogout }) {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')
  const [selectedEssay, setSelectedEssay] = useState(null)
  const [selectedQuestionSet, setSelectedQuestionSet] = useState(null)
  const [reportTab, setReportTab] = useState('feedback')

  
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
    setReportTab('feedback')
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
      <p className="eyebrow">LECTURER WORKSPACE</p>

<h1 className="dashboard-name">
  Good afternoon, {user.fullName?.split(' ')[0] || 'Lecturer'} 👋
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
          
    <section className="panel activity-chart-card">
  <div className="activity-chart-header">
    <h3>14-day activity</h3>
    <span>{summary?.recentEssays?.length ?? 0} evaluations</span>
  </div>

  <div className="activity-chart">
    <div className="chart-bars">
      {[0, 0, 0, 0, 0, 1, 0].map((value, index) => (
        <div className="chart-day" key={index}>
          <div
            className="chart-bar"
            style={{ height: value ? '140px' : '4px' }}
          />
          <small>
            {['Apr 30', 'May 2', 'May 4', 'May 6', 'May 8', 'May 10', 'May 12'][index]}
          </small>
        </div>
      ))}
    </div>
  </div>
</section>

        <section className="dashboard-stat-row">
  <article className="mini-dashboard-stat">
    <span>Essays</span>
    <strong>{summary?.totals?.essays ?? 0}</strong>
  </article>

  <article className="mini-dashboard-stat">
    <span>Sentence Checks</span>
    <strong>{summary?.totals?.sentences ?? 0}</strong>
  </article>

  <article className="mini-dashboard-stat">
    <span>Question Sets</span>
    <strong>{summary?.totals?.questionSets ?? 0}</strong>
  </article>

  <article className="mini-dashboard-stat">
    <span>Top Issue</span>
    <strong>{summary?.totals?.topCategory ?? 'None'}</strong>
  </article>
</section>

<section className="dashboard-lower-grid">

 <section className="dashboard-lower-grid">

  <div className="panel dashboard-snapshot">
    <div>
      <p className="eyebrow">ACTIVITY SNAPSHOT</p>

      <h3>Evaluation Progress</h3>

      <p>
        Your most frequent issue category is
        <strong> {summary?.totals?.topCategory ?? 'No data yet'}</strong>.
      </p>
    </div>

    <div className="snapshot-bars">
      <div>
        <span>Essays</span>
        <b style={{ width: '72%' }}></b>
      </div>

      <div>
        <span>Questions</span>
        <b style={{ width: '55%' }}></b>
      </div>

      <div>
        <span>Reports</span>
        <b style={{ width: '64%' }}></b>
      </div>
    </div>
  </div>

</section>

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
                    onClick={async () => {
                        const data = await api.get(`/questions?userId=${user.id}`)
                        const fullItem = data.items?.find((q) => q.id === item.id)
                        setSelectedQuestionSet(fullItem || item)
                    }}
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
  <div className="report-modal-overlay">
    <div className="panel report-panel stored-report-modal">

      <button
        className="modal-close"
        onClick={() => setSelectedEssay(null)}
      >
        ×
      </button>

      <div className="report-header">
        <div>
          <h2>Evaluation Report</h2>
          <p>
            {selectedEssay.summary?.overall_assessment?.overview ||
              'No overview available.'}
          </p>
        </div>

        <div className="grade-box">
          <strong>
            {selectedEssay.summary?.score < 5
              ? 'F'
              : selectedEssay.summary?.score < 6
              ? 'D'
              : selectedEssay.summary?.score < 7
              ? 'C'
              : selectedEssay.summary?.score < 8
              ? 'B'
              : 'A'}
          </strong>

          <span>
            {Number(selectedEssay.summary?.score || 7.5).toFixed(1)} / 10
          </span>
        </div>
      </div>

      <div className="tabs">
        <button
          className={reportTab === 'rubric' ? 'active' : ''}
          onClick={() => setReportTab('rubric')}
        >
          Rubric
        </button>

        <button
          className={reportTab === 'errors' ? 'active' : ''}
          onClick={() => setReportTab('errors')}
        >
          Errors (
          {Object.values(selectedEssay.summary?.counts || {}).reduce(
            (a, b) => a + b,
            0
          )}
          )
        </button>

        <button
          className={reportTab === 'feedback' ? 'active' : ''}
          onClick={() => setReportTab('feedback')}
        >
          Feedback
        </button>

        <button
          className={reportTab === 'annotated' ? 'active' : ''}
          onClick={() => setReportTab('annotated')}
        >
          Annotated
        </button>
      </div>

      <div className="tab-content">

        {reportTab === 'rubric' && (
          <div className="rubric-list">
            {[
              ['Mechanics', 8],
              ['Vocabulary', 8],
              ['Structure', 8],
              ['Grammar', 8],
              ['Content', 7.5],
            ].map(([name, value]) => (
              <div className="rubric-item" key={name}>
                <div className="rubric-title">
                  <strong>{name}</strong>
                  <span>{Number(value).toFixed(1)} / 10</span>
                </div>

                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${value * 10}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {reportTab === 'errors' && (
          <div className="errors-list">
            {selectedEssay.summary?.grouped &&
              Object.entries(selectedEssay.summary.grouped).map(
                ([category, items]) =>
                  items.map((item, index) => (
                    <div className="error-card" key={`${category}-${index}`}>
                      <div className="error-card-header">
                        <span className="error-type">
                          {category.replaceAll('_', ' ')}
                        </span>

                        <span className="severity high">
                          High
                        </span>
                      </div>

                      <div className="error-text">
                        “{item.spelling_suspects?.[0] || item.sentence}”
                      </div>

                      <p>
                        {item.reason || 'Possible issue detected.'}
                      </p>

                      <p className="suggestion">
                        → Suggest:{' '}
                        {item.suggestion ||
                          selectedEssay.summary?.tips?.[category] ||
                          'Review carefully.'}
                      </p>
                    </div>
                  ))
              )}
          </div>
        )}

        {reportTab === 'feedback' && (
          <div className="editable-feedback-box">

            <label className="feedback-label">
              Lecturer Editable Feedback
            </label>

            <textarea
              className="feedback-textarea"
              value={
                selectedEssay.lecturerEditedFeedback ||
                'No edited feedback saved yet.'
              }
              readOnly
              rows={14}
            />
          </div>
        )}

        {reportTab === 'annotated' && (
          <div className="annotated-box">
            {selectedEssay.content}
          </div>
        )}

      </div>
    </div>
  </div>
)}  
       {selectedQuestionSet && (
  <div className="report-modal-overlay">
    <div className="report-modal">

      <button
        className="modal-close"
        onClick={() => setSelectedQuestionSet(null)}
      >
        ×
      </button>

      <div className="report-header">
        <div>
          <h2>{selectedQuestionSet.title}</h2>
          <p className="muted-text">
            {selectedQuestionSet.questionCount} generated questions
          </p>
        </div>

        <div className="report-grade">
          Questions
        </div>
      </div>

      <div className="question-modal-list">
        {selectedQuestionSet.questions?.map((question, index) => (
          <div className="question-modal-card" key={`${question.type}-${index}`}>
            <strong>
              {index + 1}. {question.type}
            </strong>

            <p>{question.question}</p>

            {question.options?.length > 0 && (
              <ul>
                {question.options.map((option) => (
                  <li key={option}>{option}</li>
                ))}
              </ul>
            )}

            <small>
              Answer: {question.answer}
            </small>
          </div>
        ))}
      </div>

    </div>
  </div>
)}   
    </Layout>
  )
}
