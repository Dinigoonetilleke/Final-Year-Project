import { Link, useNavigate } from 'react-router-dom'
import AuthForm from '../components/AuthForm'
import { api } from '../lib/api'

export default function LoginPage({ form, setForm, setMessage, message, loading, setLoading, setLecturerUser, setAdminUser }) {
  const navigate = useNavigate()

  function onChange(event) {
    const { name, value } = event.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  async function onSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setMessage(null)
    try {
      const data = await api.post('/auth/login', {
        email: form.email,
        password: form.password,
      })
      if (data.user.role === 'admin') {
  setAdminUser(data.user)
  navigate('/admin')
} else {
  setLecturerUser(data.user)
  navigate('/dashboard')
}
    } catch (error) {
      setMessage({ type: 'error', text: error.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthForm
      title="User Login"
      subtitle="Login to evaluate essays, generate questions, and system manage features."
      fields={[
        { name: 'email', label: 'Email', type: 'email', placeholder: 'lecturer@example.com' },
        { name: 'password', label: 'Password', type: 'password', placeholder: 'Enter your password' },
      ]}
      form={form}
      onChange={onChange}
      onSubmit={onSubmit}
      message={message}
      loading={loading}
      submitLabel="Login"
      footer={<p className="muted-text">Need an account? <Link to="/signup">Create one now</Link>.</p>}
    />
  )
}
