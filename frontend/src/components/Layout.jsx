import { NavLink } from 'react-router-dom'

export default function Layout({ user, onLogout, title, subtitle, children, admin = false }) {
  const isAdmin = admin || user?.role === 'admin'

  const navItems = isAdmin
    ? [
        { to: '/admin', label: 'Admin Dashboard', icon: '📊' },
        { to: '/admin', label: 'Lecturers', icon: '👩‍🏫' },
        { to: '/admin', label: 'Essay Records', icon: '📝' },
        { to: '/admin', label: 'System Monitoring', icon: '⚙️' },
      ]
    : [
        { to: '/dashboard', label: 'Dashboard', icon: '🏠' },
        { to: '/essay-evaluation', label: 'Essay Evaluation', icon: '📝' },
        { to: '/question-generation', label: 'Question Generation', icon: '❔' },
        { to: '/students', label: 'Students', icon: '👥' },
        { to: '/resources', label: 'Reports & Storage', icon: '📁' },
      ]

  const initials = (user?.fullName || user?.email || 'U')
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <img src="/logo.png" alt="App Logo" className="sidebar-logo" />
          <div>
            <h1>Smart English</h1>
            <p>Essay Evaluation System</p>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.label}
              to={item.to}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <span>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p>Logged in as</p>
          <strong>{isAdmin ? 'System Admin' : 'Lecturer'}</strong>
        </div>
      </aside>

      <div className="main-area">
        <header className="app-header new-header">
          <div>
            <p className="system-label">Smart English Essay Evaluation System</p>
            <h2>{title}</h2>
            {subtitle && <p className="header-subtitle">{subtitle}</p>}
          </div>

          <div className="header-right">
            <div className="user-profile">
              <div className="avatar">{initials}</div>
              <span>{user?.fullName || user?.email}</span>
            </div>

            <button className="logout-link" onClick={onLogout}>
              Logout
            </button>
          </div>
        </header>

        <main className="main-content">{children}</main>
      </div>
    </div>
  )
}