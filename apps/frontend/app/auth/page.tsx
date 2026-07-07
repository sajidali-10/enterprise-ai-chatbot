'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { getApiBaseUrl } from '@/lib/api'
import { useAuth } from '@/contexts/AuthContext'
import { useTheme } from '@/contexts/ThemeContext'
import HipLinkLogo from '@/components/HipLinkLogo'

const DEV_USERS = [
  {
    name: 'Admin User',
    id: 'admin_user',
    role: 'admin',
    description: 'Full platform administration, document management, debug mode, observability, and evaluations.'
  },
  {
    name: 'Regular User',
    id: 'regular_user',
    role: 'user',
    description: 'Standard user with general chat, knowledge-base access, document viewing, and document upload if enabled.'
  },
  {
    name: 'Viewer User',
    id: 'viewer_user',
    role: 'viewer',
    description: 'Read-only user with knowledge-base answers, citations, feedback, and document viewing only.'
  },
]

export default function AuthPage() {
  const router = useRouter()
  const { theme, toggleTheme } = useTheme()
  const { loginWithToken, setDevUser, authMode, devAuthEnabled, configLoaded, auth } = useAuth()

  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedDevUser, setSelectedDevUser] = useState<string | null>(null)

  // Redirect already-authenticated users away from /auth
  useEffect(() => {
    if (auth?.authenticated) {
      router.replace('/')
    }
  }, [auth?.authenticated, router])

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const res = await fetch(`${getApiBaseUrl()}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username_or_email: identifier, password })
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: 'Login failed' }))
        throw new Error(data.detail || 'Invalid credentials')
      }

      const data = await res.json()
      if (!data.access_token) {
        throw new Error('Invalid response from server')
      }

      await loginWithToken(data.access_token)
      router.replace('/')
    } catch (err: any) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  function handleDevSelect(userId: string) {
    setSelectedDevUser(userId)
    setDevUser(userId)
    router.replace('/')
  }

  // Show dev mode selector only when:
  // - Backend config is loaded
  // - AUTH_MODE is 'dev' AND DEV_AUTH_ENABLED is true
  const showDevSelector = configLoaded && authMode === 'dev' && devAuthEnabled

  return (
    <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header with Logo */}
        <div className="text-center mb-8">
          <div className="inline-block mb-6">
            <HipLinkLogo variant="auto" size="lg" priority />
          </div>
          <h1 className="text-2xl font-bold text-hiplink-dark dark:text-dark-text mb-2">HipLink AI Assistant</h1>
          <p className="text-hiplink-secondary dark:text-dark-text-dim">Secure enterprise knowledge access</p>
        </div>

        {/* Login Card */}
        <div className="card dark:bg-dark-card p-6 mb-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">Sign In</h2>
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 dark:hover:bg-dark-elevated dark:text-dark-text-muted transition-colors"
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              )}
            </button>
          </div>

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-hiplink-dark dark:text-dark-text mb-1">
                Username or Email
              </label>
              <input
                type="text"
                value={identifier}
                onChange={e => setIdentifier(e.target.value)}
                required
                autoComplete="username"
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text focus:outline-none focus:ring-2 focus:ring-hiplink-blue dark:focus:ring-sky-500"
                placeholder="Enter username or email"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-hiplink-dark dark:text-dark-text mb-1">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text focus:outline-none focus:ring-2 focus:ring-hiplink-blue dark:focus:ring-sky-500"
                placeholder="Enter password"
              />
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-primary py-2.5 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Signing in…' : 'Sign In'}
            </button>
          </form>
        </div>

        {/* Dev Mode Selector (shown only when backend reports AUTH_MODE=dev) */}
        {showDevSelector && (
          <div className="card dark:bg-dark-card p-6 mb-4">
            <h3 className="text-sm font-semibold text-hiplink-secondary dark:text-dark-text-muted mb-3 uppercase tracking-wide">
              Development Mode
            </h3>
            <div className="space-y-2">
              {DEV_USERS.map(user => (
                <button
                  key={user.id}
                  onClick={() => handleDevSelect(user.id)}
                  className={`w-full text-left p-3 rounded-lg border transition-all ${
                    selectedDevUser === user.id
                      ? 'border-hiplink-blue dark:border-sky-400 bg-blue-50 dark:bg-sky-900/20'
                      : 'border-hiplink-border dark:border-dark-border hover:border-gray-300 dark:hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-hiplink-dark dark:text-dark-text text-sm">{user.name}</span>
                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                      user.role === 'admin'
                        ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400'
                        : user.role === 'user'
                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                    }`}>
                      {user.role}
                    </span>
                  </div>
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mt-1">{user.description}</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* No "Back to Home" link — / is protected and requires authentication */}
      </div>
    </main>
  )
}
