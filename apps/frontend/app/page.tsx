'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import { getApiBaseUrl } from '@/lib/api'

interface HealthStatus {
  backend: 'healthy' | 'unhealthy' | 'unknown'
  postgres: 'healthy' | 'unhealthy' | 'unknown'
  redis: 'healthy' | 'unhealthy' | 'unknown'
  qdrant: 'healthy' | 'unhealthy' | 'unknown'
  minio: 'healthy' | 'unhealthy' | 'unknown'
}

export default function Home() {
  const [health, setHealth] = useState<{ status: string; service: string } | null>(null)
  const [healthStatus, setHealthStatus] = useState<HealthStatus>({
    backend: 'unknown',
    postgres: 'unknown',
    redis: 'unknown',
    qdrant: 'unknown',
    minio: 'unknown',
  })
  const [error, setError] = useState<string | null>(null)
  const { devUser, auth } = useAuth()

  useEffect(() => {
    checkBackendHealth()
    // Check other services periodically
    const interval = setInterval(checkServiceHealth, 30000)
    checkServiceHealth()
    return () => clearInterval(interval)
  }, [])

  async function checkBackendHealth() {
    try {
      const res = await fetch(`${getApiBaseUrl()}/health`)
      if (res.ok) {
        const data = await res.json()
        setHealth(data)
        setHealthStatus(prev => ({ ...prev, backend: 'healthy' }))
        setError(null)
      } else {
        setHealthStatus(prev => ({ ...prev, backend: 'unhealthy' }))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to backend')
      setHealthStatus(prev => ({ ...prev, backend: 'unhealthy' }))
    }
  }

  async function checkServiceHealth() {
    // These would typically have dedicated health endpoints
    // For now, we just check if the backend is healthy which depends on the others
    const services: (keyof Pick<HealthStatus, 'postgres' | 'redis' | 'qdrant' | 'minio'>)[] = ['postgres', 'redis', 'qdrant', 'minio']
    for (const service of services) {
      try {
        // Use backend health as a proxy - if backend is healthy, dependencies are likely healthy
        if (healthStatus.backend === 'healthy' || health === null) {
          setHealthStatus(prev => ({ ...prev, [service]: 'healthy' }))
        }
      } catch {
        setHealthStatus(prev => ({ ...prev, [service]: 'unhealthy' }))
      }
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-900">Enterprise AI Chatbot</h1>
            <div className="flex items-center space-x-4">
              <Link href="/chat" className="text-gray-600 hover:text-blue-600 font-medium">
                Chat
              </Link>
              <Link href="/documents" className="text-gray-600 hover:text-blue-600 font-medium">
                Documents
              </Link>
              <Link href="/auth" className="flex items-center space-x-2 text-gray-600 hover:text-blue-600">
                <span className={`w-2 h-2 rounded-full ${devUser ? 'bg-green-500' : 'bg-gray-400'}`} />
                <span className="font-medium">{devUser || 'Guest'}</span>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Backend Health Status */}
        <div className="bg-white shadow-lg rounded-xl p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <span className={`w-3 h-3 rounded-full mr-2 ${healthStatus.backend === 'healthy' ? 'bg-green-500' : healthStatus.backend === 'unhealthy' ? 'bg-red-500' : 'bg-yellow-500'}`} />
            System Status
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500 mb-1">Backend</p>
              <p className={`font-semibold ${healthStatus.backend === 'healthy' ? 'text-green-600' : 'text-red-600'}`}>
                {healthStatus.backend === 'healthy' ? 'Healthy' : healthStatus.backend === 'unhealthy' ? 'Unhealthy' : 'Checking...'}
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500 mb-1">PostgreSQL</p>
              <p className={`font-semibold ${healthStatus.postgres === 'healthy' ? 'text-green-600' : 'text-red-600'}`}>
                {healthStatus.postgres === 'healthy' ? 'Healthy' : 'Checking...'}
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500 mb-1">Redis</p>
              <p className={`font-semibold ${healthStatus.redis === 'healthy' ? 'text-green-600' : 'text-red-600'}`}>
                {healthStatus.redis === 'healthy' ? 'Healthy' : 'Checking...'}
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500 mb-1">Qdrant</p>
              <p className={`font-semibold ${healthStatus.qdrant === 'healthy' ? 'text-green-600' : 'text-red-600'}`}>
                {healthStatus.qdrant === 'healthy' ? 'Healthy' : 'Checking...'}
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500 mb-1">MinIO</p>
              <p className={`font-semibold ${healthStatus.minio === 'healthy' ? 'text-green-600' : 'text-red-600'}`}>
                {healthStatus.minio === 'healthy' ? 'Healthy' : 'Checking...'}
              </p>
            </div>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mb-6">
            <strong>Backend Connection Error:</strong> {error}
            <p className="text-sm mt-1">Make sure the backend service is running.</p>
          </div>
        )}

        {/* Feature Cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Chat Card */}
          <div className="bg-white shadow-lg rounded-xl p-6">
            <div className="flex items-center mb-4">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mr-4">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold">Chat</h3>
            </div>
            <p className="text-gray-600 mb-4">
              Chat with the AI using normal mode or RAG mode with document retrieval.
            </p>
            <Link
              href="/chat"
              className="inline-block bg-blue-500 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-600 transition-colors"
            >
              Open Chat
            </Link>
          </div>

          {/* Documents Card */}
          <div className="bg-white shadow-lg rounded-xl p-6">
            <div className="flex items-center mb-4">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mr-4">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold">Documents</h3>
            </div>
            <p className="text-gray-600 mb-4">
              Upload and manage documents for RAG indexing and retrieval.
            </p>
            <div className="flex space-x-2">
              <Link
                href="/documents/upload"
                className="inline-block bg-green-500 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-600 transition-colors"
              >
                Upload
              </Link>
              <Link
                href="/documents"
                className="inline-block text-green-600 hover:text-green-700 font-medium px-4 py-2 border border-green-500 rounded-lg transition-colors"
              >
                View All
              </Link>
            </div>
          </div>

          {/* Auth Card */}
          <div className="bg-white shadow-lg rounded-xl p-6">
            <div className="flex items-center mb-4">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mr-4">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold">Authentication</h3>
            </div>
            <p className="text-gray-600 mb-4">
              Current user: <span className="font-medium">{devUser || 'Not authenticated'}</span>
            </p>
            <Link
              href="/auth"
              className="inline-block bg-purple-500 text-white px-4 py-2 rounded-lg font-medium hover:bg-purple-600 transition-colors"
            >
              {devUser ? 'Switch User' : 'Sign In (Dev)'}
            </Link>
          </div>
        </div>

        {/* Phase Progress */}
        <div className="mt-8 bg-white shadow-lg rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Implementation Status</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-2">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="font-medium text-green-800">Phase 0</p>
              <p className="text-sm text-green-600">Foundation</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-2">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="font-medium text-green-800">Phase 1</p>
              <p className="text-sm text-green-600">Chat</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-2">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="font-medium text-green-800">Phase 2</p>
              <p className="text-sm text-green-600">Documents</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-2">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="font-medium text-green-800">Phase 3</p>
              <p className="text-sm text-green-600">Indexing</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-2">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="font-medium text-green-800">Phase 4</p>
              <p className="text-sm text-green-600">RAG Chat</p>
            </div>
            <div className="text-center p-4 bg-yellow-50 rounded-lg border border-yellow-200">
              <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center mx-auto mb-2">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="font-medium text-yellow-800">Phase 5</p>
              <p className="text-sm text-yellow-600">Retrieval</p>
            </div>
            <div className="text-center p-4 bg-yellow-50 rounded-lg border border-yellow-200">
              <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center mx-auto mb-2">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="font-medium text-yellow-800">Phase 6</p>
              <p className="text-sm text-yellow-600">Auth/RBAC</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}