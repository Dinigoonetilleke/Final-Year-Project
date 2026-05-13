import { Link } from 'react-router-dom'
import { useState } from 'react'

export default function AuthForm({
  title,
  subtitle,
  fields,
  form,
  onChange,
  onSubmit,
  message,
  loading,
  submitLabel,
  footer,
}) {
  const [visiblePasswords, setVisiblePasswords] = useState({})

  function togglePassword(fieldName) {
    setVisiblePasswords((prev) => ({
      ...prev,
      [fieldName]: !prev[fieldName],
    }))
  }

  function getIcon(field) {
    if (field.type === 'email') return '＠'
    if (field.type === 'password') return '🔒'
    if (field.name === 'fullName') return '👤'
    return ''
  }

  return (
    <div className="screen-shell">
      <div className="brand-bar">Smart English Essay Evaluation System</div>

      <div className="center-wrap">
        <div className="auth-card">
          <img src="/src/assets/logo.png" className="logo" alt="Smart Essay Logo" />

          <h1>{title}</h1>

          <p className="tagline">
            <span>AI-powered</span> support for essay evaluation and question generation
          </p>

          {message && <div className={`message ${message.type}`}>{message.text}</div>}

          <form className="stack-form" onSubmit={onSubmit}>
            {fields.map((field) => {
              const isPassword = field.type === 'password'
              const inputType = isPassword && visiblePasswords[field.name] ? 'text' : field.type || 'text'

              return (
                <label key={field.name}>
                  {field.label}

                  <div className="input-wrapper">
                    <span className="input-icon">{getIcon(field)}</span>

                    <input
                      name={field.name}
                      type={inputType}
                      value={form[field.name] || ''}
                      placeholder={field.placeholder}
                      onChange={onChange}
                    />

                    {isPassword && (
                      <button
                        type="button"
                        className="password-toggle icon-toggle"
                        onClick={() => togglePassword(field.name)}
                      >
                        {visiblePasswords[field.name] ? '🙈' : '👁️'}
                      </button>
                    )}
                  </div>
                </label>
              )
            })}

            <button className="primary-btn" disabled={loading} type="submit">
              {loading ? <span className="spinner"></span> : submitLabel}
            </button>
          </form>

          <div className="auth-footer">{footer}</div>

          
        </div>
      </div>

      <div className="bottom-bar" />
    </div>
  )
}