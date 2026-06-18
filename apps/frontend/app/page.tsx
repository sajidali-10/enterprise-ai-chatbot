'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { useAuth } from '@/contexts/AuthContext'
import { getApiBaseUrl } from '@/lib/api'
import { useAuthFetch } from '@/hooks/useApi'
import ProtectedRoute from '@/components/ProtectedRoute'
import AppHeader from '@/components/AppHeader'
import { normalizePermissions, type PermissionFlags } from '@/lib/permissions'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface ServiceItem {
  label: string
  status: 'healthy' | 'unhealthy' | 'unknown' | 'active' | 'disabled' | 'info' | 'pass' | 'failure' | 'warning'
  detail?: string
}

interface StatusSection {
  title: string
  items: ServiceItem[]
}

interface KPICard {
  label: string
  value: string
  color: 'green' | 'blue' | 'yellow' | 'red' | 'gray'
  icon: React.ReactNode
}

interface SystemStatus {
  gateway: {
    nginx_proxy: string
    https_active: boolean
    domain: string
  }
  application: {
    backend_api: ServiceItem
    frontend: ServiceItem
  }
  data: {
    postgres: ServiceItem
    redis: ServiceItem
    qdrant: ServiceItem
    minio: ServiceItem
  }
  ai_provider: {
    active_provider: string
    model: string
    gateway_mode: boolean
    litellm_enabled: boolean
  }
  rag_quality: {
    last_run: string | null
    total_tests: number
    passed_tests: number
    failed_tests: number
    pass_percentage: number
    status: string
  }
  documents: {
    total_documents: number
    indexed: number
    failed: number
    pending: number
    collection_name: string
    embedding_provider: string
    embedding_dimension: number
  }
  security: {
    auth_mode: string
    current_user_role: string
    total_active_users: number
    recent_failed_logins_24h: number
    recent_audit_events_24h: number
    last_admin_action: {
      action: string
      username: string
      timestamp: string
    } | null
  }
  recent_activity: {
    recent_documents: Array<{
      id: number
      filename: string
      status: string
      created_at: string
    }>
    latest_evaluation: {
      run_id: number
      created_at: string
      pass_percentage: number
      status: string
    } | null
    recent_audit_events: Array<{
      id: number
      action: string
      username: string
      status: string
      timestamp: string
    }>
    recent_feedback: Array<{
      id: number
      rating: string
      question: string
      timestamp: string
    }>
  }
  overall_healthy: boolean
  status_source_note: string
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function statusColorClass(status: ServiceItem['status']) {
  switch (status) {
    case 'healthy':
    case 'active':
    case 'pass':
      return 'bg-emerald-500'
    case 'disabled':
    case 'info':
      return 'bg-sky-500'
    case 'warning':
    case 'unknown':
      return 'bg-amber-500'
    case 'unhealthy':
    case 'failure':
      return 'bg-red-500'
    default:
      return 'bg-gray-400'
  }
}

function statusTextClass(status: ServiceItem['status']) {
  switch (status) {
    case 'healthy':
    case 'active':
    case 'pass':
      return 'text-emerald-600 dark:text-emerald-400'
    case 'disabled':
    case 'info':
      return 'text-sky-600 dark:text-sky-400'
    case 'warning':
    case 'unknown':
      return 'text-amber-600 dark:text-amber-400'
    case 'unhealthy':
    case 'failure':
      return 'text-red-600 dark:text-red-400'
    default:
      return 'text-gray-500 dark:text-gray-400'
  }
}

function kpiColorClasses(color: KPICard['color']) {
  switch (color) {
    case 'green':
      return 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400'
    case 'blue':
      return 'bg-sky-50 dark:bg-sky-900/20 border-sky-200 dark:border-sky-800 text-sky-700 dark:text-sky-400'
    case 'yellow':
      return 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400'
    case 'red':
      return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-700 dark:text-red-400'
    case 'gray':
      return 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400'
  }
}

function kpiDotColor(color: KPICard['color']) {
  switch (color) {
    case 'green': return 'bg-emerald-500'
    case 'blue': return 'bg-sky-500'
    case 'yellow': return 'bg-amber-500'
    case 'red': return 'bg-red-500'
    case 'gray': return 'bg-gray-400'
  }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------
function StatusBadge({ status, text }: { status: ServiceItem['status']; text: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${statusTextClass(status)} bg-white dark:bg-dark-elevated border border-current/20`}>
      <span className={`w-2 h-2 rounded-full ${statusColorClass(status)}`} />
      {text}
    </span>
  )
}

function ServiceHealthCard({ title, items }: StatusSection) {
  return (
    <div className="card p-5">
      <h4 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text uppercase tracking-wider mb-4">
        {title}
      </h4>
      <div className="space-y-3">
        {items.map((item) => (
          <div key={item.label} className="flex items-center justify-between">
            <span className="text-sm text-hiplink-secondary dark:text-dark-text-muted">{item.label}</span>
            <StatusBadge status={item.status} text={
              item.status === 'healthy' ? 'Healthy' :
              item.status === 'unhealthy' ? 'Unhealthy' :
              item.status === 'active' ? 'Active' :
              item.status === 'disabled' ? 'Disabled' :
              item.status === 'info' ? 'Enabled' :
              item.status === 'unknown' ? 'Checking...' : item.status
            } />
          </div>
        ))}
      </div>
    </div>
  )
}

function KPICardComponent({ label, value, color, icon }: KPICard) {
  return (
    <div className={`rounded-xl border p-4 flex items-center gap-4 ${kpiColorClasses(color)}`}>
      <div className="w-10 h-10 rounded-lg bg-white dark:bg-dark-card flex items-center justify-center shadow-sm">
        {icon}
      </div>
      <div>
        <p className="text-xs font-medium opacity-80">{label}</p>
        <p className="text-lg font-bold">{value}</p>
      </div>
    </div>
  )
}

function OperationCard({
  title, description, href, icon,
}: {
  title: string
  description: string
  href: string
  icon: React.ReactNode
}) {
  return (
    <Link href={href} className="card p-5 hover:shadow-md transition-shadow flex items-start gap-4 group">
      <div className="w-12 h-12 rounded-xl bg-blue-50 dark:bg-sky-900/20 flex items-center justify-center flex-shrink-0">
        <div className="text-hiplink-blue dark:text-sky-400">{icon}</div>
      </div>
      <div className="flex-1 min-w-0">
        <h3 className="text-base font-semibold text-hiplink-dark dark:text-dark-text group-hover:text-hiplink-blue dark:group-hover:text-sky-400 transition-colors">
          {title}
        </h3>
        <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted mt-1">{description}</p>
      </div>
      <svg className="w-5 h-5 text-hiplink-secondary dark:text-dark-text-muted flex-shrink-0 mt-1 group-hover:text-hiplink-blue dark:group-hover:text-sky-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </Link>
  )
}

function SectionTitle({ children, icon }: { children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <h3 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text flex items-center gap-2 mb-4">
      {icon && <span>{icon}</span>}
      {children}
    </h3>
  )
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------
function DashboardContent() {
  const { auth, user } = useAuth()
  const rawRole = (auth?.role ?? 'viewer') as string
  const perms: PermissionFlags = normalizePermissions(auth?.permissions, rawRole)
  const isAdmin = perms.canAccessObservability

  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [basicHealth, setBasicHealth] = useState<{ status: string; service: string } | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const authFetch = useAuthFetch()

  useEffect(() => {
    async function load() {
      try {
        // Basic health (no auth required)
        const healthRes = await fetch(`${getApiBaseUrl()}/health`)
        if (healthRes.ok) {
          setBasicHealth(await healthRes.json())
        }
      } catch (err) {
        setHealthError(err instanceof Error ? err.message : 'Connection failed')
      }

      // Admin system status
      if (isAdmin) {
        try {
          const res = await authFetch('/api/admin/system/status')
          if (res.ok) {
            const data = await res.json()
            setSystemStatus(data)
          } else if (res.status === 403) {
            setStatusError('Admin access required for system status.')
          } else {
            setStatusError(`Failed to load system status (HTTP ${res.status}).`)
          }
        } catch (err) {
          setStatusError(err instanceof Error ? err.message : 'Failed to load system status.')
        }
      }

      setLoading(false)
    }
    load()
  }, [isAdmin, authFetch])

  const displayName = user?.full_name || user?.username || user?.email || auth?.username || ''

  // Fallback health status when system status not loaded
  const fallbackServices: StatusSection[] = [
    {
      title: 'Data Services',
      items: [
        { label: 'PostgreSQL DB', status: 'healthy' },
        { label: 'Redis Cache', status: 'healthy' },
        { label: 'Qdrant Vector DB', status: 'healthy' },
        { label: 'MinIO Object Storage', status: 'healthy' },
      ],
    },
  ]

  // Build sections from system status
  const serviceSections: StatusSection[] = systemStatus
    ? [
        {
          title: 'Gateway',
          items: [
            { label: 'Nginx Reverse Proxy', status: systemStatus.gateway.nginx_proxy === 'healthy' ? 'healthy' : 'unhealthy' },
            { label: 'HTTPS / TLS', status: systemStatus.gateway.https_active ? 'active' : 'unhealthy' },
            { label: 'Domain', status: 'info' },
          ],
        },
        {
          title: 'Application',
          items: [
            { label: systemStatus.application.backend_api.label, status: systemStatus.application.backend_api.status as ServiceItem['status'] },
            { label: systemStatus.application.frontend.label, status: systemStatus.application.frontend.status as ServiceItem['status'] },
          ],
        },
        {
          title: 'Data',
          items: [
            { label: systemStatus.data.postgres.label, status: systemStatus.data.postgres.status as ServiceItem['status'] },
            { label: systemStatus.data.redis.label, status: systemStatus.data.redis.status as ServiceItem['status'] },
            { label: systemStatus.data.qdrant.label, status: systemStatus.data.qdrant.status as ServiceItem['status'] },
            { label: systemStatus.data.minio.label, status: systemStatus.data.minio.status as ServiceItem['status'] },
          ],
        },
        {
          title: 'AI Provider',
          items: [
            { label: 'Active Provider', status: 'info' },
            { label: 'LiteLLM', status: systemStatus.ai_provider.litellm_enabled ? 'active' : 'disabled' },
            { label: 'Gateway Mode', status: systemStatus.ai_provider.gateway_mode ? 'active' : 'disabled' },
          ],
        },
      ]
    : fallbackServices

  const kpis: KPICard[] = systemStatus
    ? [
        {
          label: 'Overall Health',
          value: systemStatus.overall_healthy ? 'Healthy' : 'Degraded',
          color: systemStatus.overall_healthy ? 'green' : 'red',
          icon: (
            <svg className={`w-5 h-5 ${systemStatus.overall_healthy ? 'text-emerald-500' : 'text-red-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ),
        },
        {
          label: 'HTTPS Active',
          value: systemStatus.gateway.https_active ? 'Yes' : 'No',
          color: systemStatus.gateway.https_active ? 'green' : 'yellow',
          icon: (
            <svg className={`w-5 h-5 ${systemStatus.gateway.https_active ? 'text-emerald-500' : 'text-amber-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          ),
        },
        {
          label: 'RAG Eval',
          value: `${systemStatus.rag_quality.passed_tests}/${systemStatus.rag_quality.total_tests} (${systemStatus.rag_quality.pass_percentage}%)`,
          color: systemStatus.rag_quality.pass_percentage >= 80 ? 'green' : systemStatus.rag_quality.pass_percentage >= 50 ? 'yellow' : 'red',
          icon: (
            <svg className="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          ),
        },
        {
          label: 'Active Provider',
          value: systemStatus.ai_provider.active_provider.charAt(0).toUpperCase() + systemStatus.ai_provider.active_provider.slice(1),
          color: 'blue',
          icon: (
            <svg className="w-5 h-5 text-sky-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          ),
        },
        {
          label: 'Documents Indexed',
          value: String(systemStatus.documents.indexed),
          color: 'green',
          icon: (
            <svg className="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          ),
        },
      ]
    : basicHealth
      ? [
          {
            label: 'Backend Status',
            value: basicHealth.status === 'healthy' ? 'Healthy' : 'Unhealthy',
            color: basicHealth.status === 'healthy' ? 'green' : 'red',
            icon: (
              <svg className="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            ),
          },
        ]
      : []

  return (
    <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg">
      <AppHeader />

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Hero */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div className="flex items-center gap-4">
            <Image src="/hiplink-logo.png" alt="HipLink" width={64} height={64} className="object-contain" />
            <div>
              <h1 className="text-2xl font-bold text-hiplink-dark dark:text-dark-text">HipLink AI Assistant</h1>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-sm text-hiplink-secondary dark:text-dark-text-muted">
                  {systemStatus ? systemStatus.gateway.domain : window.location.hostname}
                </span>
                {systemStatus && (
                  <StatusBadge
                    status={systemStatus.overall_healthy ? 'healthy' : 'unhealthy'}
                    text={systemStatus.overall_healthy ? 'All Systems Healthy' : 'Systems Degraded'}
                  />
                )}
                {!systemStatus && basicHealth && (
                  <StatusBadge
                    status={basicHealth.status === 'healthy' ? 'healthy' : 'unhealthy'}
                    text={basicHealth.status === 'healthy' ? 'Backend Online' : 'Backend Unreachable'}
                  />
                )}
              </div>
            </div>
          </div>
          <div className="text-sm text-hiplink-secondary dark:text-dark-text-muted">
            Welcome back, <span className="font-medium text-hiplink-dark dark:text-dark-text">{displayName}</span>
          </div>
        </div>

        {/* KPI Row */}
        {kpis.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
            {kpis.map((kpi) => (
              <KPICardComponent key={kpi.label} {...kpi} />
            ))}
          </div>
        )}

        {/* Error Banner */}
        {healthError && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 px-4 py-3 rounded-lg mb-8">
            <strong>Backend Connection Error:</strong> {healthError}
          </div>
        )}

        {/* Admin Status Error Banner */}
        {statusError && (
          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400 px-4 py-3 rounded-lg mb-8">
            <strong>System Status:</strong> {statusError}
          </div>
        )}

        {/* Service Health */}
        <SectionTitle
          icon={
            <svg className="w-5 h-5 text-hiplink-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
        >
          Service Health
        </SectionTitle>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {serviceSections.map((section) => (
            <ServiceHealthCard key={section.title} {...section} />
          ))}
        </div>

        {/* Tooltip note */}
        {systemStatus && (
          <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted mb-8 -mt-4 italic">
            {systemStatus.status_source_note}
          </p>
        )}

        {/* RAG Quality & Documents (side by side for admins) */}
        {systemStatus && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* RAG Quality */}
            <div className="card p-5">
              <h4 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text uppercase tracking-wider mb-4 flex items-center gap-2">
                <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                RAG Quality
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Pass Rate</p>
                  <p className="text-xl font-bold text-emerald-600 dark:text-emerald-400">{systemStatus.rag_quality.pass_percentage}%</p>
                </div>
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Total Cases</p>
                  <p className="text-xl font-bold text-hiplink-dark dark:text-dark-text">{systemStatus.rag_quality.total_tests}</p>
                </div>
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Passed</p>
                  <p className="text-xl font-bold text-emerald-600 dark:text-emerald-400">{systemStatus.rag_quality.passed_tests}</p>
                </div>
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Failed</p>
                  <p className="text-xl font-bold text-red-600 dark:text-red-400">{systemStatus.rag_quality.failed_tests}</p>
                </div>
              </div>
              {systemStatus.rag_quality.last_run && (
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted mt-3">
                  Last run: {new Date(systemStatus.rag_quality.last_run).toLocaleString()}
                </p>
              )}
            </div>

            {/* Documents Summary */}
            <div className="card p-5">
              <h4 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text uppercase tracking-wider mb-4 flex items-center gap-2">
                <svg className="w-4 h-4 text-hiplink-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Documents & Indexing
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Total</p>
                  <p className="text-xl font-bold text-hiplink-dark dark:text-dark-text">{systemStatus.documents.total_documents}</p>
                </div>
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Indexed</p>
                  <p className="text-xl font-bold text-emerald-600 dark:text-emerald-400">{systemStatus.documents.indexed}</p>
                </div>
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Failed</p>
                  <p className="text-xl font-bold text-red-600 dark:text-red-400">{systemStatus.documents.failed}</p>
                </div>
                <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                  <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Pending</p>
                  <p className="text-xl font-bold text-amber-600 dark:text-amber-400">{systemStatus.documents.pending}</p>
                </div>
              </div>
              <div className="mt-3 space-y-1">
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">
                  Collection: <span className="font-medium text-hiplink-dark dark:text-dark-text">{systemStatus.documents.collection_name}</span>
                </p>
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">
                  Embedding: <span className="font-medium text-hiplink-dark dark:text-dark-text">{systemStatus.documents.embedding_provider} ({systemStatus.documents.embedding_dimension}D)</span>
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Security Summary */}
        {systemStatus && (
          <div className="card p-5 mb-8">
            <h4 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text uppercase tracking-wider mb-4 flex items-center gap-2">
              <svg className="w-4 h-4 text-hiplink-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
              Security & Admin
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Auth Mode</p>
                <p className="text-sm font-semibold text-hiplink-dark dark:text-dark-text capitalize">{systemStatus.security.auth_mode}</p>
              </div>
              <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Your Role</p>
                <p className="text-sm font-semibold text-hiplink-dark dark:text-dark-text capitalize">{systemStatus.security.current_user_role}</p>
              </div>
              <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Active Users</p>
                <p className="text-sm font-semibold text-hiplink-dark dark:text-dark-text">{systemStatus.security.total_active_users}</p>
              </div>
              <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Failed Logins (24h)</p>
                <p className={`text-sm font-semibold ${systemStatus.security.recent_failed_logins_24h > 0 ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                  {systemStatus.security.recent_failed_logins_24h}
                </p>
              </div>
              <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Audit Events (24h)</p>
                <p className="text-sm font-semibold text-hiplink-dark dark:text-dark-text">{systemStatus.security.recent_audit_events_24h}</p>
              </div>
              <div className="bg-hiplink-background dark:bg-dark-elevated rounded-lg p-3">
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">Last Admin Action</p>
                <p className="text-sm font-semibold text-hiplink-dark dark:text-dark-text truncate">
                  {systemStatus.security.last_admin_action
                    ? `${systemStatus.security.last_admin_action.action}`
                    : 'N/A'}
                </p>
                {systemStatus.security.last_admin_action?.timestamp && (
                  <p className="text-[10px] text-hiplink-secondary dark:text-dark-text-muted">
                    {new Date(systemStatus.security.last_admin_action.timestamp).toLocaleString()}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Operations */}
        <SectionTitle
          icon={
            <svg className="w-5 h-5 text-hiplink-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
          }
        >
          Operations
        </SectionTitle>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          <OperationCard
            title="Chat"
            description="Chat with the AI using general mode or RAG-powered knowledge base."
            href="/chat"
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            }
          />

          {perms.canViewDocuments && (
            <OperationCard
              title="Documents"
              description="Upload, manage, and index documents for RAG retrieval."
              href="/documents"
              icon={
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              }
            />
          )}

          {isAdmin && (
            <>
              <OperationCard
                title="Observability"
                description="Monitor usage, latency, feedback, and blocked answers."
                href="/admin/observability"
                icon={
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                }
              />

              <OperationCard
                title="Evaluations"
                description="Track controlled RAG quality tests and failure analysis."
                href="/admin/evaluations"
                icon={
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                }
              />

              <OperationCard
                title="Users"
                description="Manage user accounts, roles, and permissions."
                href="/admin/users"
                icon={
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                  </svg>
                }
              />

              <OperationCard
                title="Audit Logs"
                description="Review security audit events and user actions."
                href="/admin/audit-logs"
                icon={
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                }
              />
            </>
          )}
        </div>

        {/* Recent Activity */}
        {systemStatus && systemStatus.recent_activity && (
          <>
            <SectionTitle
              icon={
                <svg className="w-5 h-5 text-hiplink-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              }
            >
              Recent Activity
            </SectionTitle>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Recent Documents */}
              <div className="card p-5">
                <h4 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text mb-3">Recent Documents</h4>
                {systemStatus.recent_activity.recent_documents.length > 0 ? (
                  <ul className="space-y-2">
                    {systemStatus.recent_activity.recent_documents.map((doc) => (
                      <li key={doc.id} className="flex items-center justify-between text-sm">
                        <span className="truncate max-w-[200px] text-hiplink-dark dark:text-dark-text">{doc.filename}</span>
                        <StatusBadge status={doc.status === 'indexed' ? 'healthy' : doc.status === 'failed' ? 'unhealthy' : 'warning'} text={doc.status} />
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted">No recent documents.</p>
                )}
              </div>

              {/* Recent Audit Events */}
              <div className="card p-5">
                <h4 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text mb-3">Recent Audit Events</h4>
                {systemStatus.recent_activity.recent_audit_events.length > 0 ? (
                  <ul className="space-y-2">
                    {systemStatus.recent_activity.recent_audit_events.map((evt) => (
                      <li key={evt.id} className="flex items-center justify-between text-sm">
                        <span className="truncate max-w-[200px] text-hiplink-dark dark:text-dark-text">{evt.action}</span>
                        <span className="text-xs text-hiplink-secondary dark:text-dark-text-muted">{evt.username}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted">No recent events.</p>
                )}
              </div>
            </div>
          </>
        )}
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
