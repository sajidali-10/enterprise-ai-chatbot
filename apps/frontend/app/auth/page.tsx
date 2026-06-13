'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { getApiBaseUrl } from '@/lib/api'

const DEV_USERS = [
  { name: 'Admin User', id: 'admin_user', role: 'admin' },
  { name: 'Regular User', id: 'regular_user', role: 'user' },
  { name: 'Viewer User', id: 'viewer_user', role: 'viewer' },
]

export default function AuthPage() {
  const [selectedUser, setSelectedUser] = useState<string | null>(null)
  const [authInfo, setAuthInfo] = useState<any>(null)
  const [loading, setLoading] = useState(false)

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
    <main className="min-h-screen bg-hiplink-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header with Logo */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-block mb-6">
            <Image
              src="/hiplink-logo.png"
              alt="HipLink"
              width={72}
              height={72}
              className="object-contain mx-auto"
            />
          </Link>
          <h1 className="text-2xl font-bold text-hiplink-dark mb-2">HipLink AI Assistant</h1>
          <p className="text-hiplink-secondary">Secure enterprise knowledge access</p>
        </div>

        {/* Main Card */}
        <div className="card p-6 mb-4">
          <h2 className="text-lg font-semibold mb-4">Select Development User</h2>
          <p className="text-sm text-hiplink-secondary mb-4">
            In development mode, select a user to simulate authentication.
          </p>

          <div className="space-y-3">
            {DEV_USERS.map(user => (
              <button
                key={user.id}
                onClick={() => handleSelectUser(user.id)}
                className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                  selectedUser === user.id
                    ? 'border-hiplink-blue bg-blue-50'
                    : 'border-hiplink-border hover:border-gray-300 bg-white'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-hiplink-dark">{user.name}</p>
                    <p className="text-sm text-hiplink-secondary">Role: {user.role}</p>
                  </div>
                  {selectedUser === user.id && (
                    <div className="w-5 h-5 rounded-full bg-hiplink-blue flex items-center justify-center">
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>

          <button
            onClick={handleClearAuth}
            className="mt-4 w-full py-2 px-4 rounded-lg border border-hiplink-border text-hiplink-dark font-medium hover:bg-gray-50 transition-colors"
          >
            Clear Authentication
          </button>
        </div>

        {/* Auth Status */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold mb-4">Current Auth Status</h2>
          {loading ? (
            <p className="text-hiplink-secondary">Loading...</p>
          ) : authInfo ? (
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className={`w-3 h-3 rounded-full ${authInfo.authenticated ? 'bg-hiplink-success' : 'bg-gray-400'}`} />
                <span className="font-medium">
                  {authInfo.authenticated ? `Authenticated as ${authInfo.username}` : 'Not authenticated'}
                </span>
              </div>
              <p className="text-sm text-hiplink-secondary">Role: {authInfo.role}</p>
              {authInfo.user_id && <p className="text-sm text-hiplink-secondary">User ID: {authInfo.user_id}</p>}
              <p className="text-sm text-hiplink-secondary">Admin: {authInfo.is_admin ? 'Yes' : 'No'}</p>
              <p className="text-sm text-hiplink-secondary">Dev Mode Available: {authInfo.dev_mode ? 'Yes' : 'No'}</p>
            </div>
          ) : (
            <p className="text-hiplink-secondary">No auth info available</p>
          )}

          <button
            onClick={fetchAuthInfo}
            className="mt-4 w-full btn-primary"
          >
            Refresh Auth Status
          </button>
        </div>

        {selectedUser && (
          <div className="mt-4 p-4 bg-blue-50 border border-hiplink-blue rounded-lg">
            <p className="text-sm text-hiplink-blue">
              <strong>Note:</strong> The <code className="bg-blue-100 px-1 rounded">X-Dev-User: {selectedUser}</code> header will be 
              sent with API requests to simulate authentication.
            </p>
          </div>
        )}

        <div className="text-center mt-6">
          <Link href="/" className="text-hiplink-blue hover:text-hiplink-blue-dark font-medium">
            &larr; Back to Home
          </Link>
        </div>
      </div>
    </main>
  )
}