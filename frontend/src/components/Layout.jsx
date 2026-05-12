import { NavLink } from 'react-router-dom'

export default function Layout({ user, onLogout, title, subtitle, children, admin = false }) {
  const navItems = admin
    ? [{ to: '/admin', label: 'Admin Dashboard' }]
    : [
        { to: '/dashboard', label: 'Dashboard' },
        { to: '/essay-evaluation', label: 'Essay Evaluation' },
        { to: '/question-generation', label: 'Question Generation' },
        { to: '/students', label: 'Students' },
        { to: '/resources', label: 'Reports & Storage' },
      ]

  const initials = (user?.fullName || user?.email || 'U')
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()

  return (
    <div className="page-shell">
      <header className="app-header">
        <div className="header-left">
          <img src="/logo.png" alt="App Logo" className="app-logo" />

          <div>
            <p className="system-label">Smart English Essay Evaluation System</p>
            <h2>{title}</h2>
            {subtitle && <p className="header-subtitle">{subtitle}</p>}
          </div>
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

      <nav className="sub-nav clean-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `clean-nav-link ${isActive ? 'active' : ''}`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <main>{children}</main>
    </div>
  )
}

