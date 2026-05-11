import { Navigate } from 'react-router-dom'

export default function ProtectedRoute({ user, role = 'lecturer', children }) {
  if (!user) {
    return <Navigate to={role === 'admin' ? '/admin-login' : '/login'} replace />
  }
  if (role && user.role !== role) {
    return <Navigate to={user.role === 'admin' ? '/admin' : '/dashboard'} replace />
  }
  return children
}
