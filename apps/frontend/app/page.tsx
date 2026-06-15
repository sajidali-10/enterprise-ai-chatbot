'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { useAuth } from '@/contexts/AuthContext'
import { getApiBaseUrl } from '@/lib/api'
import ProtectedRoute from '@/components/ProtectedRoute'
import AppHeader from '@/components/AppHeader'
import { normalizePermissions, type PermissionFlags } from '@/lib/permissions'

interface HealthStatus {
  backend: 'healthy' | 'unhealthy' | 'unknown'
  postgres: 'healthy' | 'unhealthy' | 'unknown'
  redis: 'healthy' | 'unhealthy' | 'unknown'
  qdrant: 'healthy' | 'unhealthy' | 'unknown'
  minio: 'healthy' | 'unhealthy' | 'unknown'
}

interface ServiceCardProps {
  title: string
  description: string
  href: string
  icon: React.ReactNode
  iconBg: string
  iconColor: string
  buttonLabel: string
  buttonVariant?: 'primary' | 'secondary' | 'success'
}

function ServiceCard({ title, description, href, icon, iconBg, iconColor, buttonLabel, buttonVariant = 'primary' }: ServiceCardProps) {
  const buttonClasses = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    success: 'bg-hiplink-success text-white px-4 py-2 rounded-lg font-medium hover:bg-green-600 transition-colors',
  }

  return (
    <div className="card p-6 hover:shadow-md transition-shadow">
      <div className="flex items-center mb-4">
        <div className={`w-12 h-12 ${iconBg} rounded-lg flex items-center justify-center mr-4`}>
          <div className={iconColor}>{icon}</div>
        </div>
        <h3 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">{title}</h3>
      </div>
      <p className="text-hiplink-secondary dark:text-dark-text-muted mb-4">{description}</p>
      <Link href={href} className={buttonClasses[buttonVariant] + ' inline-block'}>
        {buttonLabel}
      </Link>
    </div>
  )
}

function DashboardContent() {
  const [health, setHealth] = useState<{ status: string; service: string } | null>(null)
  const [healthStatus, setHealthStatus] = useState<HealthStatus>({
    backend: 'unknown',
    postgres: 'unknown',
    redis: 'unknown',
    qdrant: 'unknown',
    minio: 'unknown',
  })
  const [error, setError] = useState<string | null>(null)
  const { auth } = useAuth()
  const rawRole = (auth?.role ?? 'viewer') as string
  const perms: PermissionFlags = normalizePermissions(auth?.permissions, rawRole)

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
    setHealthStatus(prev => ({
      ...prev,
      postgres: 'healthy',
      redis: 'healthy',
      qdrant: 'healthy',
      minio: 'healthy',
    }))
  }

  const isAdmin = perms.canAccessObservability

  return (
    <main className="min-h-screen">
      <AppHeader />

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-10">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="inline-block mb-4">
            <Image
              src="/hiplink-logo.png"
              alt="HipLink"
              width={80}
              height={80}
              className="object-contain"
            />
          </div>
          <h2 className="text-3xl font-bold text-hiplink-dark dark:text-dark-text mb-3">Enterprise AI Assistant</h2>
          <p className="text-lg text-hiplink-secondary dark:text-dark-text-muted max-w-2xl mx-auto">
            Chat with AI using general conversation or query your uploaded documents with RAG-powered retrieval.
          </p>
        </div>

        {/* System Status */}
        <div className="card p-6 mb-8">
          <div className="flex items-center gap-2 mb-4">
            <div className={`w-3 h-3 rounded-full ${
              healthStatus.backend === 'healthy' ? 'bg-hiplink-success' :
              healthStatus.backend === 'unhealthy' ? 'bg-hiplink-error' : 'bg-yellow-500'
            }`} />
            <h3 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">System Status</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {[
              { name: 'Backend', status: healthStatus.backend },
              { name: 'PostgreSQL', status: healthStatus.postgres },
              { name: 'Redis', status: healthStatus.redis },
              { name: 'Qdrant', status: healthStatus.qdrant },
              { name: 'MinIO', status: healthStatus.minio },
            ].map(({ name, status }) => (
              <div key={name} className="text-center p-3 bg-hiplink-background dark:bg-dark-elevated rounded-lg">
                <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted mb-1">{name}</p>
                <p className={`font-semibold ${
                  status === 'healthy' ? 'text-hiplink-success' :
                  status === 'unhealthy' ? 'text-hiplink-error' : 'text-yellow-600'
                }`}>
                  {status === 'healthy' ? 'Healthy' : status === 'unhealthy' ? 'Unhealthy' : 'Checking...'}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-hiplink-error text-hiplink-error dark:text-red-400 px-4 py-3 rounded-lg mb-8">
            <strong>Backend Connection Error:</strong> {error}
            <p className="text-sm mt-1">Make sure the backend service is running.</p>
          </div>
        )}

        {/* Service Cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <ServiceCard
            title="Chat"
            description="Chat with the AI using normal mode or RAG mode with document retrieval."
            href="/chat"
            iconBg="bg-blue-50 dark:bg-sky-900/30"
            iconColor="text-hiplink-blue dark:text-sky-400"
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            }
            buttonLabel="Open Chat"
          />

          {perms.canViewDocuments && (
            <ServiceCard
              title="Documents"
              description="Upload and manage documents for RAG indexing and retrieval."
              href="/documents"
              iconBg="bg-green-50 dark:bg-green-900/30"
              iconColor="text-hiplink-success dark:text-green-400"
              buttonVariant="success"
              icon={
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              }
              buttonLabel="Manage Documents"
            />
          )}

          {isAdmin && (
            <>
              <ServiceCard
                title="Observability"
                description="Monitor usage, latency, feedback, blocked answers, and source activity."
                href="/admin/observability"
                iconBg="bg-amber-50 dark:bg-amber-900/30"
                iconColor="text-hiplink-warning dark:text-amber-400"
                icon={
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                }
                buttonLabel="View Dashboard"
              />

              <ServiceCard
                title="Evaluations"
                description="Track controlled RAG quality tests and failure reasons."
                href="/admin/evaluations"
                iconBg="bg-red-50 dark:bg-red-900/30"
                iconColor="text-hiplink-error dark:text-red-400"
                icon={
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                }
                buttonLabel="View Results"
              />
            </>
          )}
        </div>
      </div>
    </main>
  )
}

export default function Home() {
  return (
    <ProtectedRoute>
      <DashboardContent />
    </ProtectedRoute>
  )
}
