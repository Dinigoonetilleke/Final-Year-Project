import { Link } from 'react-router-dom'

const features = [
  {
    icon: '🎯',
    title: 'Calibrated Rubric Scoring',
    desc: 'Evaluate essays using grammar, spelling, punctuation, coherence, structure, and content-based feedback.',
  },
  {
    icon: '✨',
    title: 'Verifiable Error Detection',
    desc: 'Identify common writing errors and provide structured suggestions lecturers can review before sharing.',
  },
  {
    icon: '❓',
    title: 'Comprehension Questions',
    desc: 'Generate Bloom-aligned MCQ, short-answer, and true/false questions from uploaded passages.',
  },
  {
    icon: '📋',
    title: 'Lecturer Dashboard',
    desc: 'Manage essay reports, generated questions, student performance insights, and saved materials.',
  },
]

export default function LandingPage() {
  return (
    <div className="premium-landing">
      <header className="premium-header">
        <div className="premium-container premium-nav">
          <Link to="/" className="premium-brand">
            <span className="brand-icon">🎓</span>
            <span>Smart English Essay Evaluation System</span>
          </Link>

          <div className="premium-nav-actions">
            <Link to="/login" className="premium-btn outline">Sign In</Link>
            <Link to="/signup" className="premium-btn primary">Get Started</Link>
          </div>
        </div>
      </header>

      <main>
        <section className="premium-hero">
          <div className="premium-container">
            <div className="hero-badge">✦ Built for university English lecturers</div>

            <h1>
              Assess English essays with structured feedback support.
            </h1>

            <p>
              A lecturer-focused web application that evaluates student essays,
              detects grammar, spelling, punctuation, and sentence-structure issues,
              and supports comprehension question generation.
            </p>

            <div className="premium-hero-actions">
              <Link to="/signup" className="premium-btn primary large">
                Start evaluating →
              </Link>
              <a href="#features" className="premium-btn outline large">
                See how it works
              </a>
            </div>

            <div className="premium-highlights">
              <span>✓ Essay evaluation</span>
              <span>✓ Error detection</span>
              <span>✓ Bloom-aligned questions</span>
            </div>
          </div>
        </section>

        <section id="features" className="premium-features">
          <div className="premium-container">
            <div className="section-heading">
              <h2>Everything you need to assess writing.</h2>
              <p>An academic assistant designed to support lecturers, not replace them.</p>
            </div>

            <div className="premium-feature-grid">
              {features.map((feature) => (
                <div className="premium-feature-card" key={feature.title}>
                  <div className="feature-icon">{feature.icon}</div>
                  <h3>{feature.title}</h3>
                  <p>{feature.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="premium-cta">
          <div className="premium-container">
            <h2>Spend less time marking. Spend more time teaching.</h2>
            <Link to="/signup" className="premium-btn primary large">
              Get started →
            </Link>
          </div>
        </section>
      </main>
    </div>
  )
}