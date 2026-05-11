function CountBadge({ label, value }) {
  return (
    <div className="count-badge">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default function EvaluationPage({ user, essayTitle, essayText, setEssayTitle, setEssayText, onSubmit, result, loading, onBack }) {
  return (
    <div className="page-shell">
      <header className="top-nav">
        <div>
          <h2>Essay Evaluation System</h2>
          <p>Essay Submission</p>
        </div>
        <div className="top-nav-actions">
          <span className="user-pill">{user.fullName}</span>
          <button className="ghost-btn" onClick={onBack}>
            Back to Dashboard
          </button>
        </div>
      </header>

      <main className="evaluate-layout">
        <section className="panel submission-panel">
          <h1>Submit Your Essay</h1>
          <div className="stack-form">
            <label>
              Essay Title
              <input value={essayTitle} onChange={(e) => setEssayTitle(e.target.value)} placeholder="Enter a title" />
            </label>

            <label>
              Essay Text
              <textarea
                value={essayText}
                onChange={(e) => setEssayText(e.target.value)}
                placeholder="Paste your essay here..."
                rows={12}
              />
            </label>

            <button className="primary-btn" onClick={onSubmit} disabled={loading}>
              {loading ? 'Evaluating...' : 'Submit Essay'}
            </button>
          </div>
        </section>

        <section className="panel result-panel">
          <h3>Structured Feedback</h3>
          {!result ? (
            <p className="muted-text">Submit an essay to see categorized feedback and counts.</p>
          ) : (
            <>
              <div className="badge-grid">
                {Object.entries(result.counts).map(([key, value]) => (
                  <CountBadge key={key} label={key.replaceAll('_', ' ')} value={value} />
                ))}
              </div>

              <div className="result-groups">
                {Object.entries(result.grouped).map(([label, items]) => (
                  <div className="feedback-block" key={label}>
                    <h4>{label.replaceAll('_', ' ')}</h4>
                    <p className="tip-text">{result.tips[label]}</p>
                    {items.map((item) => (
                      <div className="feedback-item" key={`${label}-${item.sentence_no}`}>
                        <strong>Sentence {item.sentence_no}</strong>
                        <p>{item.sentence}</p>
                        {item.spelling_suspects?.length > 0 && (
                          <small>Suspicious words: {item.spelling_suspects.join(', ')}</small>
                        )}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  )
}
