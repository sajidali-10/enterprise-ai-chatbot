'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
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

  // Refetch auth info when selectedUser changes
  useEffect(() => {
    fetchAuthInfo()
  }, [selectedUser])

  function handleSelectUser(userId: string) {
    setSelectedUser(userId)
    // Set in localStorage for persistence
    localStorage.setItem('dev_user', userId)
  }

  function handleClearAuth() {
    setSelectedUser(null)
    localStorage.removeItem('dev_user')
  }

  useEffect(() => {
    // Check localStorage on mount and set selectedUser if found
    const stored = localStorage.getItem('dev_user')
    if (stored && !selectedUser) {
      setSelectedUser(stored)
    } else if (!stored && !selectedUser) {
      // No stored user, fetch auth info with no dev user
      fetchAuthInfo()
    }
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold">Dev Authentication</h1>
          <Link href="/" className="text-blue-500 hover:text-blue-600 font-medium">
            Back to Home
          </Link>
        </div>

        <div className="bg-white shadow-lg rounded-lg p-6 mb-4">
          <h2 className="text-lg font-semibold mb-4">Select Development User</h2>
          <p className="text-sm text-gray-500 mb-4">
            In development mode, select a user to simulate authentication.
          </p>

          <div className="space-y-3">
            {DEV_USERS.map(user => (
              <button
                key={user.id}
                onClick={() => handleSelectUser(user.id)}
                className={`w-full text-left p-4 rounded-lg border-2 transition-colors ${
                  selectedUser === user.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{user.name}</p>
                    <p className="text-sm text-gray-500">Role: {user.role}</p>
                  </div>
                  {selectedUser === user.id && (
                    <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center">
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
            className="mt-4 w-full py-2 px-4 rounded-lg border border-gray-300 text-gray-700 font-medium hover:bg-gray-50 transition-colors"
          >
            Clear Authentication
          </button>
        </div>

        {/* Auth Status */}
        <div className="bg-white shadow-lg rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Current Auth Status</h2>
          {loading ? (
            <p className="text-gray-500">Loading...</p>
          ) : authInfo ? (
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className={`w-3 h-3 rounded-full ${authInfo.authenticated ? 'bg-green-500' : 'bg-gray-400'}`} />
                <span className="font-medium">
                  {authInfo.authenticated ? `Authenticated as ${authInfo.username}` : 'Not authenticated'}
                </span>
              </div>
              <p className="text-sm text-gray-600">Role: {authInfo.role}</p>
              {authInfo.user_id && <p className="text-sm text-gray-600">User ID: {authInfo.user_id}</p>}
              <p className="text-sm text-gray-600">Admin: {authInfo.is_admin ? 'Yes' : 'No'}</p>
              <p className="text-sm text-gray-600">Dev Mode Available: {authInfo.dev_mode ? 'Yes' : 'No'}</p>
            </div>
          ) : (
            <p className="text-gray-500">No auth info available</p>
          )}

          <button
            onClick={fetchAuthInfo}
            className="mt-4 w-full py-2 px-4 rounded-lg bg-blue-500 text-white font-medium hover:bg-blue-600 transition-colors"
          >
            Refresh Auth Status
          </button>
        </div>

        {selectedUser && (
          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-700">
              <strong>Note:</strong> The <code className="bg-blue-100 px-1 rounded">X-Dev-User: {selectedUser}</code> header will be 
              sent with API requests to simulate authentication.
            </p>
          </div>
        )}
      </div>
    </main>
  )
}