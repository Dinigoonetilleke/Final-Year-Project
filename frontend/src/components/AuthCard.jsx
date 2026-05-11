import { useState } from 'react'

export default function AuthCard({
  mode,
  form = { fullName: '', email: '', password: '', confirmPassword: '' },
  onChange,
  onSubmit,
  onSwitch,
  message,
  loading,
}) {
  const isLogin = mode === 'login'
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)

  return (
    <div className="screen-shell auth-shell">
      <div className="brand-bar">Essay Evaluation System</div>
      <div className="center-wrap">
        <div className="auth-card">
          <div className="shield-badge">🛡️</div>
          <h1>{isLogin ? 'Lecturer Login' : 'Sign Up'}</h1>
          <p className="muted-text">
            {isLogin
              ? 'Enter your account details to continue.'
              : 'Create a lecturer account to manage essay evaluations.'}
          </p>

          <form onSubmit={onSubmit} className="stack-form">
            {!isLogin && (
              <label>
                Full Name
                <input
                  type="text"
                  name="fullName"
                  placeholder="Enter your full name"
                  value={form.fullName || ''}
                  onChange={onChange}
                />
              </label>
            )}

            <label>
              Email
              <input
                type="email"
                name="email"
                placeholder="Enter your email"
                value={form.email || ''}
                onChange={onChange}
              />
            </label>

            <label>
              Password
              <div className="password-wrapper">
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  placeholder="Enter your password"
                  value={form.password || ''}
                  onChange={onChange}
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? 'Hide' : 'Show'}
                </button>
              </div>
            </label>

            {!isLogin && (
              <label>
                Confirm Password
                <div className="password-wrapper">
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    name="confirmPassword"
                    placeholder="Confirm your password"
                    value={form.confirmPassword || ''}
                    onChange={onChange}
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? 'Hide' : 'Show'}
                  </button>
                </div>
              </label>
            )}

            {message && <div className={`message ${message.type}`}>{message.text}</div>}

            <button className="primary-btn" type="submit" disabled={loading}>
              {loading ? 'Please wait...' : isLogin ? 'Login' : 'Sign Up'}
            </button>
          </form>

          <button className="text-link" type="button" onClick={onSwitch}>
            {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Login'}
          </button>
        </div>
      </div>
      <div className="bottom-bar" />
    </div>
  )
}