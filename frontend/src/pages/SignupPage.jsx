import { Link, useNavigate } from 'react-router-dom'
import AuthForm from '../components/AuthForm'
import { api } from '../lib/api'

export default function SignupPage({ form, setForm, setMessage, message, loading, setLoading }) {
  const navigate = useNavigate()

  function onChange(event) {
    const { name, value } = event.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  async function onSubmit(event) {
    event.preventDefault()
    if (!form.fullName || !form.email || !form.password || !form.confirmPassword) {
      setMessage({ type: 'error', text: 'Please fill in all fields.' })
      return
    }
    if (form.password !== form.confirmPassword) {
      setMessage({ type: 'error', text: 'Passwords do not match.' })
      return
    }

    setLoading(true)
    setMessage(null)
    try {
      await api.post('/auth/register', {
        fullName: form.fullName,
        email: form.email,
        password: form.password,
      })
      setMessage({ type: 'success', text: 'Account created. Please login.' })
      setForm({ fullName: '', email: form.email, password: '', confirmPassword: '' })
      navigate('/login')
    } catch (error) {
      setMessage({ type: 'error', text: error.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthForm
      title="Lecturer Sign Up"
      subtitle="Create your lecturer account to access essay evaluation and question generation."
      fields={[
        { name: 'fullName', label: 'Full Name', placeholder: 'Enter your full name' },
        { name: 'email', label: 'Email', type: 'email', placeholder: 'lecturer@example.com' },
        { name: 'password', label: 'Password', type: 'password', placeholder: 'Minimum 6 characters' },
        { name: 'confirmPassword', label: 'Confirm Password', type: 'password', placeholder: 'Re-enter password' },
      ]}
      form={form}
      onChange={onChange}
      onSubmit={onSubmit}
      message={message}
      loading={loading}
      submitLabel="Create Account"
      footer={
        <p className="muted-text">
            Already registered?{' '}
            <Link className="auth-link" to="/login">
                Go to login
            </Link>
        </p>
    }
    />
  )
}
