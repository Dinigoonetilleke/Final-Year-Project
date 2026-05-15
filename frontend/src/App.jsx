import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import ProtectedRoute from './components/ProtectedRoute'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import SignupPage from './pages/SignupPage'
import DashboardPage from './pages/DashboardPage'
import EssayEvaluationPage from './pages/EssayEvaluationPage'
import QuestionGenerationPage from './pages/QuestionGenerationPage'
import ResourcesPage from './pages/ResourcesPage'
import ReportsStoragePage from './pages/ReportsStoragePage'
import AdminDashboardPage from './pages/AdminDashboardPage'
import StudentsPage from './pages/StudentsPage'
import StudentDetailPage from './pages/StudentDetailPage'
import './styles.css'

const blankForm = { fullName: '', email: '', password: '', confirmPassword: '' }

export default function App() {
  const [lecturerUser, setLecturerUser] = useState(() => {
    const saved = localStorage.getItem('essay-user')
    return saved ? JSON.parse(saved) : null
  })

  const [adminUser, setAdminUser] = useState(() => {
    const saved = localStorage.getItem('essay-admin')
    return saved ? JSON.parse(saved) : null
  })

  const [form, setForm] = useState(blankForm)
  const [message, setMessage] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (lecturerUser) localStorage.setItem('essay-user', JSON.stringify(lecturerUser))
    else localStorage.removeItem('essay-user')
  }, [lecturerUser])

  useEffect(() => {
    if (adminUser) localStorage.setItem('essay-admin', JSON.stringify(adminUser))
    else localStorage.removeItem('essay-admin')
  }, [adminUser])

  function logoutLecturer() {
  localStorage.removeItem('essay-user')

  setLecturerUser(null)
  setMessage(null)

  window.location.href = '/'
}

function logoutAdmin() {
  localStorage.removeItem('essay-admin')

  setAdminUser(null)
  setMessage(null)

  window.location.href = '/'
}

  return (
    <Routes>
      <Route path="/" element={<LandingPage/>} />

      <Route
        path="/login"
        element={
          <LoginPage
            form={form}
            setForm={setForm}
            setMessage={setMessage}
            message={message}
            loading={loading}
            setLoading={setLoading}
            setLecturerUser={setLecturerUser}
            setAdminUser={setAdminUser}
          />
        }
      />

      <Route
        path="/signup"
        element={
          <SignupPage
            form={form}
            setForm={setForm}
            setMessage={setMessage}
            message={message}
            loading={loading}
            setLoading={setLoading}
          />
        }
      />

      <Route
        path="/dashboard"
        element={
          <ProtectedRoute user={lecturerUser}>
            <DashboardPage user={lecturerUser} onLogout={logoutLecturer} />
          </ProtectedRoute>
        }
      />

      <Route
        path="/essay-evaluation"
        element={
          <ProtectedRoute user={lecturerUser}>
            <EssayEvaluationPage user={lecturerUser} onLogout={logoutLecturer} />
          </ProtectedRoute>
        }
      />

      <Route
        path="/question-generation"
        element={
          <ProtectedRoute user={lecturerUser}>
            <QuestionGenerationPage user={lecturerUser} onLogout={logoutLecturer} />
          </ProtectedRoute>
        }
      />

      
    <Route
        path="/resources"
        element={
            <ProtectedRoute user={lecturerUser}>
                <ReportsStoragePage user={lecturerUser} onLogout={logoutLecturer} />
            </ProtectedRoute>
        }
    />
    
    <Route
        path="/students"
        element={
            <ProtectedRoute user={lecturerUser}>
                <StudentsPage user={lecturerUser} onLogout={logoutLecturer} />
            </ProtectedRoute>
        }
    />
          
      <Route
		path="/admin"
		element={
			<AdminDashboardPage
				user={adminUser}
				onLogout={logoutAdmin}
			/>
		}
	/>
          
      <Route
        path="/students/:id"
        element={
            <ProtectedRoute user={lecturerUser}>
                <StudentDetailPage user={lecturerUser} onLogout={logoutLecturer} />
            </ProtectedRoute>
        }
    />
          
    </Routes>
      
    
      
      
  )
}