import { useState } from 'react'
import Layout from '../components/Layout'
import { api } from '../lib/api'

export default function QuestionGenerationPage({ user, onLogout }) {
  const [title, setTitle] = useState('Generated Question Set')
  const [text, setText] = useState('')
  const [questionTypes, setQuestionTypes] = useState(['mcq', 'short', 'true_false'])
  const [result, setResult] = useState(null)
  const [message, setMessage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [openAnswers, setOpenAnswers] = useState({})
  const [count, setCount] = useState(12)
        
  function toggleType(type) {
    setQuestionTypes((prev) =>
      prev.includes(type) ? prev.filter((item) => item !== type) : [...prev, type]
    )
  }

  function toggleAnswer(index) {
    setOpenAnswers((prev) => ({ ...prev, [index]: !prev[index] }))
  }

  async function handleGenerate(event) {
    event.preventDefault()
    setLoading(true)
    setMessage(null)

    try {
      const data = await api.post('/questions/generate', { title, text, questionTypes, count, })
      setResult(data.result)
      setOpenAnswers({})
      setMessage({ type: 'success', text: 'Questions generated successfully.' })
    } catch (error) {
      setMessage({ type: 'error', text: error.message })
    } finally {
      setLoading(false)
    }
  }

  async function saveQuestions() {
    if (!result?.questions?.length) {
      setMessage({ type: 'error', text: 'Please generate questions before saving.' })
      return
    }

    try {
      await api.post('/questions/save', {
        title,
        sourceText: text,
        questions: result.questions,
        userId: user.id,
      })

      setMessage({ type: 'success', text: 'Question set saved successfully.' })
    } catch (error) {
      setMessage({ type: 'error', text: error.message })
    }
  }

  return (
    <Layout
      user={user}
      onLogout={onLogout}
      title="Question Generation"
      subtitle="Generate Bloom-aligned comprehension questions from a passage."
    >
      <div className="evaluate-layout">
        <section className="panel submission-panel">
          <h1>Create Questions</h1>

          {message && <div className={`message ${message.type}`}>{message.text}</div>}

          <form className="stack-form" onSubmit={handleGenerate}>
            <label>
              Question Set Title
              <input value={title} onChange={(e) => setTitle(e.target.value)} />
            </label>

            <label>
              Passage
              <textarea
                rows="16"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste passage or lesson text here..."
              />
            </label>
              
            <label>
                Number of Questions
                <input
                    type="number"
                    value={count}
                    min={5}
                    max={20}
                    onChange={(e) => setCount(Number(e.target.value))}
                />
            </label>

            <div className="checkbox-group">
              {[
                ['mcq', 'MCQ'],
                ['short', 'Short Answer'],
                ['true_false', 'True / False'],
              ].map(([value, label]) => (
                <label key={value} className="checkbox-pill">
                  <input
                    type="checkbox"
                    checked={questionTypes.includes(value)}
                    onChange={() => toggleType(value)}
                  />
                  {label}
                </label>
              ))}
            </div>

            <button className="primary-btn" disabled={loading} type="submit">
              {loading ? 'Generating...' : 'Generate Questions'}
            </button>
          </form>
        </section>

        <section className="panel">
          <h3>Generated Output</h3>

          {!result ? (
            <p className="muted-text">Generated questions will appear here.</p>
          ) : (
            <div className="result-stack">
              <div className="action-row">
                <button className="primary-btn" onClick={saveQuestions}>
                  Save Question Set
                </button>
              </div>

              {result.questions.map((question, index) => (
                <div className="question-card" key={`${question.type}-${index}`}>
                  <div className="question-meta">
                    <span>Q{index + 1}</span>
                    <span>{question.type}</span>
                    <span>{question.difficulty || 'Medium'}</span>
                    <span>{question.bloom || 'Understand'}</span>
                  </div>

                  <p className="question-text">{question.question}</p>

                  {question.options && (
                    <ol className="option-list" type="A">
                      {question.options.map((option) => (
                        <li key={option}>{option}</li>
                      ))}
                    </ol>
                  )}

                  <button
                    type="button"
                    className="answer-toggle"
                    onClick={() => toggleAnswer(index)}
                  >
                    {openAnswers[index] ? 'Hide answer' : 'Show answer'}
                  </button>

                  {openAnswers[index] && (
                    <div className="answer-box">
                      <strong>Answer:</strong> {question.answer}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </Layout>
  )
}