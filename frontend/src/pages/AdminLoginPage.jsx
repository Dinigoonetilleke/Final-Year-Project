import { useNavigate } from 'react-router-dom'
import AuthForm from '../components/AuthForm'
import { api } from '../lib/api'

export default function AdminLoginPage({ adminForm, setAdminForm, setAdminUser, setMessage, message, loading, setLoading }) {
  const navigate = useNavigate()

  function onChange(event) {
    const { name, value } = event.target
    setAdminForm((prev) => ({ ...prev, [name]: value }))
  }

  async function onSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setMessage(null)
    try {
      const data = await api.post('/auth/admin/login', {
        email: adminForm.email,
        password: adminForm.password,
      })
      setAdminUser(data.user)
      navigate('/admin')
    } catch (error) {
      setMessage({ type: 'error', text: error.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthForm
      title="Admin Login"
      subtitle="Use the admin account to view system-wide users, essays, and question sets."
      fields={[
        { name: 'email', label: 'Admin Email', type: 'email', placeholder: 'admin@smartessay.local' },
        { name: 'password', label: 'Password', type: 'password', placeholder: 'Enter admin password' },
      ]}
      form={adminForm}
      onChange={onChange}
      onSubmit={onSubmit}
      message={message}
      loading={loading}
      submitLabel="Admin Login"
      footer={<p className="muted-text">Default admin: <strong>admin@smartessay.local</strong> / <strong>Admin123!</strong></p>}
    />
  )
}
