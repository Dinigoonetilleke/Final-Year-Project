import { useEffect, useState } from 'react'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function ReportsStoragePage({ user, onLogout }) {
  const [reports, setReports] = useState([])
  const [questions, setQuestions] = useState([])
  const [selectedReport, setSelectedReport] = useState(null)
  const [selectedQuestionSet, setSelectedQuestionSet] = useState(null)

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
    setSelectedReport(data)
    setSelectedQuestionSet(null)
  }

  function openQuestionSet(item) {
    setSelectedQuestionSet(item)
    setSelectedReport(null)
  }

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

          {reports.length ? (
            reports.map((report) => (
              <div
                className="row-item"
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

          {questions.length ? (
            questions.map((item) => (
              <div
                className="row-item"
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
        <section className="panel" style={{ marginTop: '24px' }}>
          <h2>{selectedReport.title}</h2>

          <div className="feedback-item">
            <h4>Essay Text</h4>
            <p>{selectedReport.content}</p>
          </div>

          <div className="feedback-item">
            <h4>Overall Feedback</h4>
            <p>{selectedReport.summary?.overall_assessment?.overview}</p>

            {selectedReport.summary?.overall_assessment?.improvements?.length > 0 && (
              <ul>
                {selectedReport.summary.overall_assessment.improvements.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            )}
          </div>

          <div className="feedback-item">
            <h4>Essay Metrics</h4>
            <p>
              Readability: {selectedReport.summary?.essay_metrics?.readabilityScore} (
              {selectedReport.summary?.essay_metrics?.readabilityLevel})
            </p>
            <p>
              Average sentence length:{' '}
              {selectedReport.summary?.average_sentence_length} words
            </p>
            <p>
              Lexical diversity:{' '}
              {selectedReport.summary?.essay_metrics?.lexicalDiversity}
            </p>
          </div>

          <div className="feedback-item">
            <h4>Issue Counts</h4>
            <div className="chips-wrap">
              {Object.entries(selectedReport.summary?.counts || {}).map(([key, value]) => (
                <span className="chip" key={key}>
                  {key.replaceAll('_', ' ')}: {value}
                </span>
              ))}
            </div>
          </div>

          <div className="feedback-item">
            <h4>Detailed Sentence Feedback</h4>

            {Object.keys(selectedReport.summary?.grouped || {}).length ? (
              Object.entries(selectedReport.summary.grouped).map(([category, items]) => (
                <div key={category} className="feedback-group">
                  <h5>{category.replaceAll('_', ' ')}</h5>

                  {items.map((item) => (
                    <div key={`${category}-${item.sentence_no}`} className="sentence-feedback">
                      <p>
                        <b>Sentence {item.sentence_no}:</b> {item.sentence}
                      </p>
                      <p>
                        <b>Reason:</b> {item.reason || 'Possible issue detected.'}
                      </p>
                      <p>
                        <b>Suggestion:</b>{' '}
                        {item.suggestion || 'Review this sentence carefully.'}
                      </p>
                    </div>
                  ))}
                </div>
              ))
            ) : (
              <p className="muted-text">No detailed sentence issues detected.</p>
            )}
          </div>
        </section>
      )}

      {selectedQuestionSet && (
        <section className="panel" style={{ marginTop: '24px' }}>
          <h2>{selectedQuestionSet.title}</h2>
          <p className="muted-text">
            {selectedQuestionSet.questionCount} questions generated
          </p>

          {selectedQuestionSet.questions?.map((question, index) => (
            <div className="feedback-item" key={`${question.type}-${index}`}>
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

              <small>Answer: {question.answer}</small>
            </div>
          ))}
        </section>
      )}
    </Layout>
  )
}