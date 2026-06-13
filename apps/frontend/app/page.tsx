'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
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
    const services: (keyof Pick<HealthStatus, 'postgres' | 'redis' | 'qdrant' | 'minio'>)[] = ['postgres', 'redis', 'qdrant', 'minio']
    for (const service of services) {
      try {
        if (healthStatus.backend === 'healthy' || health === null) {
          setHealthStatus(prev => ({ ...prev, [service]: 'healthy' }))
        }
      } catch {
        setHealthStatus(prev => ({ ...prev, [service]: 'unhealthy' }))
      }
    }
  }

  return (
    <main className="min-h-screen">
      {/* Navigation Bar */}
      <nav className="brand-header">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Image
                src="/hiplink-logo.png"
                alt="HipLink"
                width={36}
                height={36}
                className="object-contain"
              />
              <h1 className="text-xl font-semibold text-hiplink-dark">HipLink AI Assistant</h1>
            </div>
            <div className="flex items-center space-x-6">
              <Link href="/chat" className="text-hiplink-secondary hover:text-hiplink-blue font-medium transition-colors">
                Chat
              </Link>
              <Link href="/documents" className="text-hiplink-secondary hover:text-hiplink-blue font-medium transition-colors">
                Documents
              </Link>
              <Link href="/auth" className="flex items-center space-x-2 text-hiplink-secondary hover:text-hiplink-blue">
                <span className={`w-2 h-2 rounded-full ${devUser ? 'bg-hiplink-success' : 'bg-gray-400'}`} />
                <span className="font-medium">{devUser || 'Guest'}</span>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Backend Health Status */}
        <div className="card p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center">
            <span className={`w-3 h-3 rounded-full mr-2 ${healthStatus.backend === 'healthy' ? 'bg-hiplink-success' : healthStatus.backend === 'unhealthy' ? 'bg-hiplink-error' : 'bg-yellow-500'}`} />
            System Status
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {[
              { name: 'Backend', status: healthStatus.backend },
              { name: 'PostgreSQL', status: healthStatus.postgres },
              { name: 'Redis', status: healthStatus.redis },
              { name: 'Qdrant', status: healthStatus.qdrant },
              { name: 'MinIO', status: healthStatus.minio },
            ].map(({ name, status }) => (
              <div key={name} className="text-center p-4 bg-hiplink-background rounded-lg">
                <p className="text-sm text-hiplink-secondary mb-1">{name}</p>
                <p className={`font-semibold ${status === 'healthy' ? 'text-hiplink-success' : status === 'unhealthy' ? 'text-hiplink-error' : 'text-yellow-600'}`}>
                  {status === 'healthy' ? 'Healthy' : status === 'unhealthy' ? 'Unhealthy' : 'Checking...'}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 border border-hiplink-error text-hiplink-error px-4 py-3 rounded-lg mb-6">
            <strong>Backend Connection Error:</strong> {error}
            <p className="text-sm mt-1">Make sure the backend service is running.</p>
          </div>
        )}

        {/* Feature Cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Chat Card */}
          <div className="card p-6">
            <div className="flex items-center mb-4">
              <div className="w-12 h-12 bg-blue-50 rounded-lg flex items-center justify-center mr-4">
                <svg className="w-6 h-6 text-hiplink-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold">Chat</h3>
            </div>
            <p className="text-hiplink-secondary mb-4">
              Chat with the AI using normal mode or RAG mode with document retrieval.
            </p>
            <Link
              href="/chat"
              className="btn-primary inline-block"
            >
              Open Chat
            </Link>
          </div>

          {/* Documents Card */}
          <div className="card p-6">
            <div className="flex items-center mb-4">
              <div className="w-12 h-12 bg-green-50 rounded-lg flex items-center justify-center mr-4">
                <svg className="w-6 h-6 text-hiplink-success" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold">Documents</h3>
            </div>
            <p className="text-hiplink-secondary mb-4">
              Upload and manage documents for RAG indexing and retrieval.
            </p>
            <div className="flex space-x-2">
              <Link
                href="/documents/upload"
                className="bg-hiplink-success text-white px-4 py-2 rounded-lg font-medium hover:bg-green-600 transition-colors inline-block"
              >
                Upload
              </Link>
              <Link
                href="/documents"
                className="text-hiplink-success hover:text-green-700 font-medium px-4 py-2 border border-hiplink-success rounded-lg transition-colors inline-block"
              >
                View All
              </Link>
            </div>
          </div>

          {/* Auth Card */}
          <div className="card p-6">
            <div className="flex items-center mb-4">
              <div className="w-12 h-12 bg-purple-50 rounded-lg flex items-center justify-center mr-4">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold">Authentication</h3>
            </div>
            <p className="text-hiplink-secondary mb-4">
              Current user: <span className="font-medium">{devUser || 'Not authenticated'}</span>
            </p>
            <Link
              href="/auth"
              className="bg-purple-500 text-white px-4 py-2 rounded-lg font-medium hover:bg-purple-600 transition-colors inline-block"
            >
              {devUser ? 'Switch User' : 'Sign In (Dev)'}
            </Link>
          </div>
        </div>
      </div>
    </main>
  )
}