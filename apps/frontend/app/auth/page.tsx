'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { getApiBaseUrl } from '@/lib/api'
import { useTheme } from '@/contexts/ThemeContext'

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
  const [selectedUser, setSelectedUser] = useState<string | null>(null)
  const [authInfo, setAuthInfo] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const { theme, toggleTheme } = useTheme()

  useEffect(() => {
    fetchAuthInfo()
  }, [])

  function fetchAuthInfo() {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (selectedUser) {
      headers['X-Dev-User'] = selectedUser
    }
    setLoading(true)
    fetch(`${getApiBaseUrl()}/api/chat/auth-info`, { headers })
      .then(res => res.json())
      .then(data => {
        setAuthInfo(data)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to fetch auth info:', err)
        setLoading(false)
      })
  }

  useEffect(() => {
    fetchAuthInfo()
  }, [selectedUser])

  function handleSelectUser(userId: string) {
    setSelectedUser(userId)
    localStorage.setItem('dev_user', userId)
  }

  function handleClearAuth() {
    setSelectedUser(null)
    localStorage.removeItem('dev_user')
  }

  useEffect(() => {
    const stored = localStorage.getItem('dev_user')
    if (stored && !selectedUser) {
      setSelectedUser(stored)
    } else if (!stored && !selectedUser) {
      fetchAuthInfo()
    }
  }, [])

  return (
    <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header with Logo */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-block mb-6">
            <Image
              src="/hiplink-logo.png"
              alt="HipLink"
              width={120}
              height={120}
              className="object-contain mx-auto"
            />
          </Link>
          <h1 className="text-2xl font-bold text-hiplink-dark dark:text-dark-text mb-2">HipLink AI Assistant</h1>
          <p className="text-hiplink-secondary dark:text-dark-text-dim">Secure enterprise knowledge access</p>
        </div>

        {/* Main Card */}
        <div className="card dark:bg-dark-card p-6 mb-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">Select Development User</h2>
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
          <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim mb-4">
            In development mode, select a user to simulate authentication.
          </p>

          <div className="space-y-3">
            {DEV_USERS.map(user => (
              <button
                key={user.id}
                onClick={() => handleSelectUser(user.id)}
                className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                  selectedUser === user.id
                    ? 'border-hiplink-blue dark:border-sky-400 bg-blue-50 dark:bg-sky-900/20'
                    : 'border-hiplink-border dark:border-dark-border hover:border-gray-300 dark:hover:border-slate-600 bg-white dark:bg-dark-card'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-hiplink-dark dark:text-dark-text">{user.name}</p>
                    <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">{user.description}</p>
                  </div>
                  {selectedUser === user.id && (
                    <div className="w-6 h-6 rounded-full bg-hiplink-blue dark:bg-sky-500 flex items-center justify-center flex-shrink-0">
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  )}
                </div>
                <div className="mt-1">
                  <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                    user.role === 'admin' 
                      ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400' 
                      : user.role === 'user'
                      ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                  }`}>
                    {user.role}
                  </span>
                </div>
              </button>
            ))}
          </div>

          <button
            onClick={handleClearAuth}
            className="mt-4 w-full py-2.5 px-4 rounded-lg border border-hiplink-border dark:border-dark-border text-hiplink-dark dark:text-dark-text font-medium hover:bg-gray-50 dark:hover:bg-dark-elevated transition-colors"
          >
            Clear Authentication
          </button>
        </div>

        {/* Auth Status */}
        <div className="card dark:bg-dark-card p-6">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-4">Current Auth Status</h2>
          {loading ? (
            <p className="text-hiplink-secondary dark:text-dark-text-dim">Loading...</p>
          ) : authInfo ? (
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <span className={`w-3 h-3 rounded-full ${authInfo.authenticated ? 'bg-hiplink-success' : 'bg-gray-400'}`} />
                <span className="font-medium text-hiplink-dark dark:text-dark-text">
                  {authInfo.authenticated ? `Authenticated as ${authInfo.username}` : 'Not authenticated'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-2">
                  <span className="text-hiplink-secondary dark:text-dark-text-dim">Role:</span>
                  <span className="font-medium text-hiplink-dark dark:text-dark-text ml-1">{authInfo.role}</span>
                </div>
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-2">
                  <span className="text-hiplink-secondary dark:text-dark-text-dim">Admin:</span>
                  <span className="font-medium text-hiplink-dark dark:text-dark-text ml-1">{authInfo.is_admin ? 'Yes' : 'No'}</span>
                </div>
              </div>
              {authInfo.user_id && (
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-2">
                  <span className="text-hiplink-secondary dark:text-dark-text-dim text-xs">User ID:</span>
                  <span className="font-mono text-hiplink-dark dark:text-dark-text ml-1 text-xs">{authInfo.user_id}</span>
                </div>
              )}
              <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-2">
                <span className="text-hiplink-secondary dark:text-dark-text-dim text-xs">Dev Mode:</span>
                <span className="font-medium text-hiplink-dark dark:text-dark-text ml-1 text-xs">{authInfo.dev_mode ? 'Available' : 'Not available'}</span>
              </div>
            </div>
          ) : (
            <p className="text-hiplink-secondary dark:text-dark-text-dim">No auth info available</p>
          )}

          <button
            onClick={fetchAuthInfo}
            className="mt-4 w-full btn-primary"
          >
            Refresh Auth Status
          </button>
        </div>

        {selectedUser && (
          <div className="mt-4 p-4 bg-blue-50 dark:bg-sky-900/20 border border-hiplink-blue dark:border-sky-600 rounded-lg">
            <p className="text-sm text-hiplink-blue dark:text-sky-400">
              <strong>Note:</strong> The <code className="bg-blue-100 dark:bg-sky-900/50 px-1 rounded">X-Dev-User: {selectedUser}</code> header will be 
              sent with API requests to simulate authentication.
            </p>
          </div>
        )}

        {/* Development mode notice */}
        <div className="mt-4 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-300 dark:border-amber-700 rounded-lg">
          <p className="text-sm text-amber-800 dark:text-amber-300">
            <strong>Development Mode:</strong> This is a role simulation for development purposes. Production authentication and user management will be added in Phase 12.
          </p>
        </div>

        <div className="text-center mt-6">
          <Link href="/" className="text-hiplink-blue dark:text-sky-400 hover:text-hiplink-blue-dark dark:hover:text-sky-300 font-medium flex items-center justify-center gap-1">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Home
          </Link>
        </div>
      </div>
    </main>
  )
}