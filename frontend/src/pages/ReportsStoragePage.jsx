import { useEffect, useState } from 'react'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function ReportsStoragePage({ user, onLogout }) {
  const [reports, setReports] = useState([])
  const [questions, setQuestions] = useState([])
  const [selectedReport, setSelectedReport] = useState(null)
  const [selectedQuestionSet, setSelectedQuestionSet] = useState(null)
  const [reportSearch, setReportSearch] = useState('')
  const [questionSearch, setQuestionSearch] = useState('')
  const [reportTab, setReportTab] = useState('summary')

  useEffect(() => {
    loadStorage()
  }, [user.id])

  async function loadStorage() {
    const reportData = await api.get(`/reports?userId=${user.id}`)
    const questionData = await api.get(`/questions?userId=${user.id}`)

    setReports(reportData.reports || [])
    setQuestions(questionData.items || [])
  }

  async function openReport(id) {
    const data = await api.get(`/reports/${id}`)
    setSelectedReport(data.report || data)
    setReportTab('feedback')
    setSelectedQuestionSet(null)
  }

  function openQuestionSet(item) {
    setSelectedQuestionSet(item)
    setSelectedReport(null)
  }

  const filteredReports = reports.filter((report) =>
  `${report.title} ${report.rating}`
    .toLowerCase()
    .includes(reportSearch.toLowerCase())
  )

  const filteredQuestions = questions.filter((item) =>
  `${item.title}`
    .toLowerCase()
    .includes(questionSearch.toLowerCase())
  )

  return (
    <Layout
      user={user}
      onLogout={onLogout}
      title="Reports & Storage"
      subtitle="View saved essay reports and generated question sets."
    >
      <div className="evaluate-layout">
        <section className="panel">
          <h3>Saved Essay Reports</h3>
            
          <input
    	       className="storage-search"
               type="text"
               placeholder="Search essay reports..."
               value={reportSearch}
               onChange={(e) => setReportSearch(e.target.value)}
          />

          {reports.length ? (
            filteredReports.map((report) => (
              <div
                className="storage-row"
                key={report.id}
                onClick={() => openReport(report.id)}
                style={{ cursor: 'pointer' }}
              >
                <strong>{report.title}</strong>
                <p>
                  {report.wordCount} words • {report.sentenceCount} sentences •{' '}
                  {report.rating}
                </p>
                <small>{new Date(report.createdAt).toLocaleString()}</small>
              </div>
            ))
          ) : (
            <p className="muted-text">No essay reports saved yet.</p>
          )}
        </section>

        <section className="panel">
          <h3>Saved Question Sets</h3>
            
            <input
                className="storage-search"
                type="text"
                placeholder="Search question sets..."
                value={questionSearch}
                onChange={(e) => setQuestionSearch(e.target.value)}
            />

          {questions.length ? (
            filteredQuestions.map((item) => (
              <div
                className="storage-row"
                key={item.id}
                onClick={() => openQuestionSet(item)}
                style={{ cursor: 'pointer' }}
              >
                <strong>{item.title}</strong>
                <p>{item.questionCount} questions generated</p>
                <small>{new Date(item.createdAt).toLocaleString()}</small>
              </div>
            ))
          ) : (
            <p className="muted-text">No question sets saved yet.</p>
          )}
        </section>
      </div>

{selectedReport && (
  <div className="report-modal-overlay">
    <div className="panel report-panel stored-report-modal">

      <button
        className="modal-close"
        onClick={() => setSelectedReport(null)}
      >
        ×
      </button>

      <div className="report-header">
        <div>
          <h2>Evaluation Report</h2>
          <p>
            {selectedReport.summary?.overall_assessment?.overview ||
              'No overview available.'}
          </p>
        </div>

        <div className="grade-box">
          <strong>
            {selectedReport.summary?.score < 5
              ? 'F'
              : selectedReport.summary?.score < 6
              ? 'D'
              : selectedReport.summary?.score < 7
              ? 'C'
              : selectedReport.summary?.score < 8
              ? 'B'
              : 'A'}
          </strong>
          <span>
            {Number(selectedReport.summary?.score || 7.5).toFixed(1)} / 10
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
          {Object.values(selectedReport.summary?.counts || {}).reduce(
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
            {selectedReport.summary?.grouped &&
              Object.entries(selectedReport.summary.grouped).map(
                ([category, items]) =>
                  items.map((item, index) => (
                    <div className="error-card" key={`${category}-${index}`}>
                      <div className="error-card-header">
                        <span className="error-type">
                          {category.replaceAll('_', ' ')}
                        </span>
                        <span className="severity high">High</span>
                      </div>

                      <div className="error-text">
                        “{item.spelling_suspects?.[0] || item.sentence}”
                      </div>

                      <p>{item.reason || 'Possible issue detected.'}</p>

                      <p className="suggestion">
                        → Suggest:{' '}
                        {item.suggestion ||
                          selectedReport.summary?.tips?.[category] ||
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
                selectedReport.lecturerEditedFeedback ||
                'No edited feedback saved yet.'
              }
              readOnly
              rows={14}
            />
          </div>
        )}

        {reportTab === 'annotated' && (
          <div className="annotated-box">
            {selectedReport.content}
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