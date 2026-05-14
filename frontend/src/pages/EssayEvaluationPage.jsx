import { useEffect, useState } from 'react'
import Layout from '../components/Layout'
import { api } from '../lib/api'
import mammoth from 'mammoth'

export default function EssayEvaluationPage({ user, onLogout }) {
  const [title, setTitle] = useState('Student Essay Report')
  const [students, setStudents] = useState([])
  const [studentInput, setStudentInput] = useState('')
  const [selectedStudentId, setSelectedStudentId] = useState('')
  const [essay, setEssay] = useState('')
  const [result, setResult] = useState(null)
  const [editedFeedback, setEditedFeedback] = useState("")
  const [message, setMessage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [ocrLoading, setOcrLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('rubric')
  const [isOcrText, setIsOcrText] = useState(false)
  
  
  useEffect(() => {
  async function loadStudents() {
    try {
      const data = await api.get(`/students?userId=${user.id}`)
      setStudents(data.students || [])
    } catch (error) {
      console.error(error)
    }
  }

  if (user?.id) {
    loadStudents()
  }
}, [user])
  

  async function handleFileUpload(event) {
    const file = event.target.files[0]
    if (!file) return

    const extension = file.name.split('.').pop().toLowerCase()

    setResult(null)
    setIsOcrText(false)

    try {
      setMessage({
        type: 'success',
        text: 'Processing file...',
      })

      if (extension === 'txt') {
        const reader = new FileReader()
        reader.onload = (e) => {
          setEssay(e.target.result)
          setMessage({
            type: 'success',
            text: 'Text file loaded successfully.',
          })
        }
        reader.readAsText(file)
      }

      else if (extension === 'docx') {
        const arrayBuffer = await file.arrayBuffer()
        const result = await mammoth.extractRawText({ arrayBuffer })

        setEssay(result.value)
        setMessage({
          type: 'success',
          text: 'DOCX file loaded successfully.',
        })
      }
	  
	  else if (extension === 'pdf') {
		setOcrLoading(true)

		const formData = new FormData()
		formData.append('file', file)

		const res = await fetch('http://localhost:5000/api/extract-text/pdf', {
			method: 'POST',
			body: formData,
		})

		const data = await res.json()

		if (!res.ok || data.error) {
			setMessage({
				type: 'error',
				text: data.error || 'PDF extraction failed.',
			})
			return
		}

		setEssay(data.text || '')
		setIsOcrText(true)

		setMessage({
			type: 'success',
			text: 'PDF text extracted successfully.',
		})
	}

      else if (['jpg', 'jpeg', 'png'].includes(extension)) {
        setOcrLoading(true)

        const formData = new FormData()
        formData.append('file', file)

        const res = await fetch('http://localhost:5000/api/extract-text/image', {
          method: 'POST',
          body: formData,
        })

        const data = await res.json()

        if (!res.ok || data.error) {
          setMessage({
            type: 'error',
            text: data.error || 'OCR failed. Please try another image.',
          })
          return
        }

        setEssay(data.text || '')
        setIsOcrText(true)

        setMessage({
          type: 'success',
          text: '⚠️ OCR text extracted. Please correct the text before clicking Evaluate.',
        })
      }

      else {
        setMessage({
          type: 'error',
          text: 'Only .txt, .docx,pdf, .jpg, .jpeg, and .png files are allowed.',
        })
      }
    } catch (err) {
      console.error(err)
      setMessage({
        type: 'error',
        text: 'Error reading file.',
      })
    } finally {
      setOcrLoading(false)
      event.target.value = ''
    }
  }

  function looksLikeBadOcr(text) {
    const symbols = (text.match(/[^a-zA-Z0-9\s.,!?'"()-]/g) || []).length
    const words = text.trim().split(/\s+/).filter(Boolean)
    const shortWords = words.filter((word) => word.length <= 2).length

    return (
      text.length < 80 ||
      symbols > 20 ||
      (words.length > 20 && shortWords / words.length > 0.45)
    )
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setMessage(null)

    if (!essay.trim()) {
      setMessage({
        type: 'error',
        text: 'Please enter or upload an essay before evaluation.',
      })
      setLoading(false)
      return
    }

    if (looksLikeBadOcr(essay)) {
      setMessage({
        type: 'error',
        text: 'The essay text looks unclear. Please clean the OCR text before evaluating.',
      })
      setLoading(false)
      return
    }

    try {
      const selectedStudent = students.find(
        (student) => student.id === selectedStudentId
      );

      const data = await api.post('/evaluate', {
        title,
        essay,
        userId: user.id,
        studentId: selectedStudent?.id || null,
        studentName: selectedStudent?.name || null,
        studentNumber: selectedStudent?.studentNumber || null,
        batch: selectedStudent?.batch || null,
      });

setResult({
  ...data.result,
  essayId: data.essayId,
  reportId: data.reportId,
})

const rating = data.result.overall_assessment?.rating || ""

const improvements =
  data.result.overall_assessment?.improvements || []

const strengths = []

if ((data.result.counts?.Grammar || 0) <= 1) {
  strengths.push("Grammar usage is generally accurate.")
}

if ((data.result.counts?.Sentence_Structure || 0) <= 1) {
  strengths.push("Sentence structure is clear and readable.")
}

if ((data.result.counts?.Spelling || 0) === 0) {
  strengths.push("Spelling accuracy is very good.")
}

if ((data.result.counts?.Punctuation || 0) <= 1) {
  strengths.push("Punctuation has been used effectively.")
}

if (strengths.length === 0) {
  strengths.push("The essay communicates ideas adequately.")
}

let overallComment = ""

if (rating === "Excellent") {
  overallComment =
    "This is a well-written essay with strong language usage and organization."
} else if (rating === "Good") {
  overallComment =
    "The essay is clear and understandable with only a few language issues."
} else if (rating === "Satisfactory") {
  overallComment =
    "The essay communicates ideas but requires improvements in several areas."
} else {
  overallComment =
    "The essay needs significant improvement in grammar, structure, and clarity."
}

const feedbackText = [
  "Overall Feedback:",
  overallComment,
  "",
  "Strengths:",
  ...strengths.map((item) => `- ${item}`),
  "",
  "Areas for Improvement:",
  ...improvements.map((item) => `- ${item}`),
  "",
  "Lecturer Final Feedback:",
  "",
].join("\n")

setEditedFeedback(feedbackText)

setEditedFeedback(feedbackText)
setActiveTab('feedback')

      const fullReport = {
        id: crypto.randomUUID(),
        title: title || 'Untitled Essay Report',
        essayText: essay,
        result: data.result,
        createdAt: new Date().toISOString(),
      }

      const existingReports =
        JSON.parse(localStorage.getItem('essayReports')) || []

      existingReports.unshift(fullReport)
      localStorage.setItem('essayReports', JSON.stringify(existingReports))

      setMessage({
        type: 'success',
        text: 'Essay evaluated and full report saved successfully.',
      })
    } catch (error) {
      setMessage({
        type: 'error',
        text: error.message || 'Evaluation failed.',
      })
    } finally {
      setLoading(false)
    }
  }

  const totalErrors = Object.values(result?.counts || {}).reduce(
    (a, b) => a + b,
    0
  )

  const score =
    result?.score ||
    (result?.overall_assessment?.rating === 'Needs Improvement' ? 4.6 : 7.5)

  const grade =
    score < 5 ? 'F' : score < 6 ? 'D' : score < 7 ? 'C' : score < 8 ? 'B' : 'A'

  const rubric = [
    [
      'Mechanics',
      Math.max(
        2,
        10 -
          ((result?.counts?.Punctuation || 0) +
            (result?.counts?.Spelling || 0)) *
            0.8
      ),
    ],
    ['Vocabulary', Math.max(2, 10 - (result?.counts?.Word_Usage || 0) * 1.1)],
    [
      'Structure',
      Math.max(2, 10 - (result?.counts?.Sentence_Structure || 0) * 1.2),
    ],
    ['Grammar', Math.max(2, 10 - (result?.counts?.Grammar || 0) * 1.1)],
    ['Content', totalErrors > 8 ? 5.5 : 7.5],
  ]

async function handleSaveEditedFeedback() {
  if (!result?.essayId) {
    alert("Please evaluate an essay first.")
    return
  }

  try {
    await api.post(`/reports/${result.essayId}/feedback`, {
      editedFeedback,
    })

    alert("Edited feedback saved successfully.")
  } catch (error) {
    alert(error.message || "Could not save edited feedback.")
  }
}
            
  return (
    <Layout
      user={user}
      onLogout={onLogout}
      title="Evaluate an Essay"
      subtitle="Paste or upload a student essay. The system will return rubric scores, feedback, and error highlights."
    >
      <div className="evaluate-layout">
        <section className="panel submission-panel">
          <h1>Submission</h1>

          {message && (
            <div className={`message ${message.type}`}>{message.text}</div>
          )}

          <form className="stack-form" onSubmit={handleSubmit}>
              
            <label>
                Student
                <select
                    value={selectedStudentId}
                    onChange={(e) => {
                        setSelectedStudentId(e.target.value);
                    }}
                >
                    <option value="">-- Select student --</option>

                    {students.map((student) => (
                        <option key={student.id} value={student.id}>
                            {student.name} - {student.studentNumber || "No ID"}
                            {student.batch ? ` - ${student.batch}` : ""}
                        </option>
                    ))}
                </select>
            </label>
              
            <label>
              Report Title
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter report title"
              />
            </label>

            <label>
              Upload Essay File
              <input
                className="file-input"
                type="file"
                accept=".txt,.docx,.jpg,.jpeg,.png,.pdf"
                onChange={handleFileUpload}
                disabled={ocrLoading || loading}
              />
            </label>

            {ocrLoading && (
              <p className="muted-text">Extracting handwritten text...</p>
            )}

            {isOcrText && (
              <div className="message warning">
                ⚠️ Handwritten OCR may contain mistakes. Please edit the text
                below before evaluating.
              </div>
            )}

            <label>
              Essay Text
              <textarea
                rows="16"
                value={essay}
                onChange={(e) => setEssay(e.target.value)}
                placeholder="Paste essay here or upload a file. If OCR text appears, correct it before evaluation."
                required
              />
            </label>

            <button
              className="primary-btn"
              type="submit"
              disabled={loading || ocrLoading}
            >
              {loading ? 'Evaluating...' : 'Evaluate Essay'}
            </button>
          </form>
        </section>

        <section className="panel report-panel">
          {!result ? (
            <p className="muted-text">
              Your essay report will appear here after submission.
            </p>
          ) : (
            <>
              <div className="report-header">
                <div>
                  <h2>Evaluation Report</h2>
                  <p>{result.overall_assessment?.overview}</p>
                </div>

                <div className="grade-box">
                  <strong>{grade}</strong>
                  <span>{Number(score).toFixed(1)} / 10</span>
                </div>
              </div>

              <div className="tabs">
                <button
                  type="button"
                  className={activeTab === 'rubric' ? 'active' : ''}
                  onClick={() => setActiveTab('rubric')}
                >
                  Rubric
                </button>

                <button
                  type="button"
                  className={activeTab === 'errors' ? 'active' : ''}
                  onClick={() => setActiveTab('errors')}
                >
                  Errors ({totalErrors})
                </button>

                <button
                  type="button"
                  className={activeTab === 'feedback' ? 'active' : ''}
                  onClick={() => setActiveTab('feedback')}
                >
                  Feedback
                </button>

                <button
                  type="button"
                  className={activeTab === 'annotated' ? 'active' : ''}
                  onClick={() => setActiveTab('annotated')}
                >
                  Annotated
                </button>
              </div>

              <div className="tab-content">
                {activeTab === 'rubric' && (
                  <div className="rubric-list">
                    {rubric.map(([name, value]) => (
                      <div className="rubric-item" key={name}>
                        <div className="rubric-title">
                          <strong>{name}</strong>
                          <span>{value.toFixed(1)} / 10</span>
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

                {activeTab === 'errors' && (
                  <div className="errors-list">
                    {result.grouped &&
                      Object.entries(result.grouped).map(([category, items]) =>
                        items.map((item) => (
                          <div
                            className="error-card"
                            key={`${category}-${item.sentence_no}`}
                          >
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
                                result.tips?.[category] ||
                                'Review carefully.'}
                            </p>
                          </div>
                        ))
                      )}
                  </div>
                )}

                {activeTab === "feedback" && (
                    <div className="editable-feedback-box">
                        <label className="feedback-label">
                            Lecturer Editable Feedback
                        </label>

                        <textarea
                            className="feedback-textarea"
                            value={editedFeedback}
                            onChange={(e) => setEditedFeedback(e.target.value)}
                            rows={14}
                        />

                        <button
                            type="button"
                            className="save-feedback-button"
                            onClick={handleSaveEditedFeedback}
                        >
                        Save Edited Feedback
                        </button>
                </div>
            )}

                {activeTab === 'annotated' && (
                  <div className="annotated-box">{essay}</div>
                )}
              </div>
                
            </>
          )}
        </section>
      </div>
    </Layout>
  )
}