'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { useAuth } from '@/contexts/AuthContext'
import { useAuthFetch } from '@/hooks/useApi'
import ProtectedRoute from '@/components/ProtectedRoute'
import AppHeader from '@/components/AppHeader'
import { normalizePermissions, type PermissionFlags } from '@/lib/permissions'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface ServiceItem {
  label: string
  status: string
  detail?: string
}

interface StatusGroup {
  title: string
  items: ServiceItem[]
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
// Status helpers — dark-aware color tokens
// ---------------------------------------------------------------------------
function statusDotClass(status: string) {
  const s = status.toLowerCase()
  if (['healthy', 'active', 'pass', 'indexed', 'success', 'operational', 'online'].includes(s)) {
    return 'bg-emerald-500'
  }
  if (['unhealthy', 'failure', 'failed', 'error', 'degraded', 'offline'].includes(s)) {
    return 'bg-red-500'
  }
  if (['warning', 'pending', 'unknown'].includes(s)) {
    return 'bg-amber-500'
  }
  return 'bg-sky-500'
}

function statusTextClass(status: string) {
  const s = status.toLowerCase()
  if (['healthy', 'active', 'pass', 'indexed', 'success', 'operational', 'online'].includes(s)) {
    return 'text-emerald-600 dark:text-emerald-400'
  }
  if (['unhealthy', 'failure', 'failed', 'error', 'degraded', 'offline'].includes(s)) {
    return 'text-red-600 dark:text-red-400'
  }
  if (['warning', 'pending', 'unknown'].includes(s)) {
    return 'text-amber-600 dark:text-amber-400'
  }
  return 'text-sky-600 dark:text-sky-400'
}

function statusBadgeClass(status: string) {
  const s = status.toLowerCase()
  if (['healthy', 'active', 'pass', 'indexed', 'success', 'operational', 'online'].includes(s)) {
    return 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20'
  }
  if (['unhealthy', 'failure', 'failed', 'error', 'degraded', 'offline'].includes(s)) {
    return 'bg-red-50 dark:bg-red-500/10 text-red-700 dark:text-red-400 border-red-200 dark:border-red-500/20'
  }
  if (['warning', 'pending', 'unknown'].includes(s)) {
    return 'bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/20'
  }
  return 'bg-sky-50 dark:bg-sky-500/10 text-sky-700 dark:text-sky-400 border-sky-200 dark:border-sky-500/20'
}

// ---------------------------------------------------------------------------
// Skeleton components
// ---------------------------------------------------------------------------
function SkeletonRow({ w = 'w-3/4' }: { w?: string }) {
  return <div className={`h-3 bg-gray-200 dark:bg-slate-700/50 rounded ${w} animate-pulse`} />
}

function SkeletonCard() {
  return (
    <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-5 space-y-3">
      <SkeletonRow w="w-1/2" />
      <SkeletonRow w="w-3/4" />
      <SkeletonRow w="w-1/3" />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Small inline degraded indicator
// ---------------------------------------------------------------------------
function RefreshIndicator({ error }: { error: string | null }) {
  if (!error) return null
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-500/20"
      title={error}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
      Refresh failed
    </span>
  )
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------
const Icons = {
  server: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 01-2 2v4a2 2 0 012 2h14a2 2 0 012-2v-4a2 2 0 01-2-2m-2-4h.01M17 16h.01" />
    </svg>
  ),
  database: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2 3 3 8 3s8-1 8-3V7M4 7c0 2 3 3 8 3s8-1 8-3M4 7c0-2 3-3 8-3s8 1 8 3" />
    </svg>
  ),
  shield: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  ),
  document: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  ),
  users: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
    </svg>
  ),
  eye: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  ),
  checkmark: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 13l4 4L19 7" />
    </svg>
  ),
  chart: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  ),
  chat: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  ),
  lightning: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  lock: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
    </svg>
  ),
  globe: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
    </svg>
  ),
  activity: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  ),
}

// ---------------------------------------------------------------------------
// KPI Card
// ---------------------------------------------------------------------------
function KPICard({ label, value, icon, tone }: { label: string; value: string; icon: React.ReactNode; tone: 'green' | 'blue' | 'amber' | 'slate' }) {
  const toneMap = {
    green: {
      bg: 'bg-emerald-50 dark:bg-emerald-500/10',
      border: 'border-emerald-200 dark:border-emerald-500/20',
      text: 'text-emerald-700 dark:text-emerald-400',
      iconBg: 'bg-emerald-100 dark:bg-emerald-500/20',
    },
    blue: {
      bg: 'bg-sky-50 dark:bg-sky-500/10',
      border: 'border-sky-200 dark:border-sky-500/20',
      text: 'text-sky-700 dark:text-sky-400',
      iconBg: 'bg-sky-100 dark:bg-sky-500/20',
    },
    amber: {
      bg: 'bg-amber-50 dark:bg-amber-500/10',
      border: 'border-amber-200 dark:border-amber-500/20',
      text: 'text-amber-700 dark:text-amber-400',
      iconBg: 'bg-amber-100 dark:bg-amber-500/20',
    },
    slate: {
      bg: 'bg-white dark:bg-dark-card',
      border: 'border-hiplink-border dark:border-dark-border',
      text: 'text-hiplink-dark dark:text-dark-text',
      iconBg: 'bg-gray-100 dark:bg-slate-700/30',
    },
  }
  const t = toneMap[tone]
  return (
    <div className={`${t.bg} ${t.border} border rounded-xl p-4 flex items-center gap-3`}>
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${t.iconBg}`}>
        <div className={t.text}>{icon}</div>
      </div>
      <div>
        <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted">{label}</p>
        <p className="text-sm font-semibold text-hiplink-dark dark:text-dark-text">{value}</p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Service Health Group
// ---------------------------------------------------------------------------
function ServiceHealthGroup({ title, items }: StatusGroup) {
  return (
    <div className="card p-5">
      <h4 className="text-xs font-semibold text-hiplink-secondary dark:text-dark-text-muted uppercase tracking-wider mb-4">
        {title}
      </h4>
      <div className="space-y-2.5">
        {items.map((item) => (
          <div key={item.label} className="flex items-center justify-between text-sm">
            <span className="text-hiplink-dark dark:text-dark-text">{item.label}</span>
            <span className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${statusDotClass(item.status)}`} />
              <span className={`text-xs font-medium ${statusTextClass(item.status)}`}>
                {item.detail || item.status}
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Operation Card
// ---------------------------------------------------------------------------
function OperationCard({ title, description, href, icon, color }: {
  title: string; description: string; href: string; icon: React.ReactNode; color: string
}) {
  return (
    <Link
      href={href}
      className="card p-5 flex items-center gap-4 hover:shadow-md hover:border-hiplink-blue/40 dark:hover:border-hiplink-blue/30 transition-all group"
    >
      <div className={`w-11 h-11 rounded-lg flex items-center justify-center flex-shrink-0 ${color}`}>
        <div className="text-white">{icon}</div>
      </div>
      <div className="flex-1 min-w-0">
        <h3 className="text-sm font-medium text-hiplink-dark dark:text-dark-text group-hover:text-hiplink-blue dark:group-hover:text-sky-400 transition-colors">
          {title}
        </h3>
        <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mt-0.5 leading-relaxed">
          {description}
        </p>
      </div>
      <svg className="w-4 h-4 text-hiplink-secondary dark:text-dark-text-dim flex-shrink-0 group-hover:text-hiplink-blue dark:group-hover:text-sky-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </Link>
  )
}

// ---------------------------------------------------------------------------
// Dashboard Content
// ---------------------------------------------------------------------------
function DashboardContent() {
  const { auth, user, loading: authLoading } = useAuth()
  const authFetch = useAuthFetch()
  const rawRole = (auth?.role ?? 'viewer') as string
  const perms: PermissionFlags = normalizePermissions(auth?.permissions, rawRole)
  const isAdmin = perms.canAccessObservability

  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)

  // Refs to avoid dependency loops
  const authFetchRef = useRef(authFetch)
  const abortRef = useRef<AbortController | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const fetchInProgressRef = useRef(false)

  authFetchRef.current = authFetch

  /**
   * Load status — supports both initial (shows loading) and background (silent).
   * Uses refs for authFetch to keep deps stable (prevents useEffect loops).
   */
  const loadStatus = useCallback(async (isBackground = false) => {
    if (!isAdmin) {
      setLoading(false)
      return
    }
    if (fetchInProgressRef.current) return

    fetchInProgressRef.current = true
    if (!isBackground) setLoading(true)
    setStatusError(null)

    try {
      if (abortRef.current) abortRef.current.abort()
      abortRef.current = new AbortController()

      const res = await authFetchRef.current('/api/admin/system/status', {
        signal: abortRef.current.signal,
      })

      if (res.ok) {
        const data: SystemStatus = await res.json()
        setSystemStatus(data)
        setStatusError(null)
        setLastRefresh(new Date())
      } else if (res.status === 401 || res.status === 403) {
        setStatusError('Access denied')
      } else {
        setStatusError(`Status unavailable (${res.status})`)
      }
    } catch (err: any) {
      // Ignore AbortError — expected on unmount / tab-switch
      if (err?.name === 'AbortError') return
      setStatusError('Refresh failed')
    } finally {
      fetchInProgressRef.current = false
      if (!isBackground) setLoading(false)
    }
  }, [isAdmin])

  // Initial fetch + background refresh every 60s.
  // Runs once isAdmin is resolved (authLoading is false).
  useEffect(() => {
    if (authLoading) return

    loadStatus(false)

    intervalRef.current = setInterval(() => {
      loadStatus(true)
    }, 60000)

    return () => {
      if (abortRef.current) abortRef.current.abort()
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [authLoading, loadStatus])

  const displayName = user?.full_name || user?.username || user?.email || auth?.username || ''

  // -------------------------------------------------------------------------
  // Build service health groups from status or degraded defaults
  // -------------------------------------------------------------------------
  const buildGroups = (): StatusGroup[] => {
    if (!systemStatus) {
      return [
        { title: 'Application', items: [{ label: 'Frontend UI', status: 'unknown' }, { label: 'Backend API', status: 'unknown' }] },
        { title: 'Gateway', items: [{ label: 'Nginx Reverse Proxy', status: 'unknown' }, { label: 'HTTPS / TLS', status: 'unknown' }, { label: 'Domain', status: 'unknown' }] },
        { title: 'Data', items: [{ label: 'PostgreSQL DB', status: 'unknown' }, { label: 'Redis Cache', status: 'unknown' }, { label: 'Qdrant Vector DB', status: 'unknown' }, { label: 'MinIO Object Storage', status: 'unknown' }] },
        { title: 'AI', items: [{ label: 'Active Provider', status: 'unknown' }, { label: 'Model', status: 'unknown' }, { label: 'LiteLLM Gateway', status: 'unknown' }] },
      ]
    }
    return [
      {
        title: 'Application',
        items: [
          { label: systemStatus.application.frontend.label, status: systemStatus.application.frontend.status },
          { label: systemStatus.application.backend_api.label, status: systemStatus.application.backend_api.status },
        ],
      },
      {
        title: 'Gateway',
        items: [
          { label: 'Nginx Reverse Proxy', status: systemStatus.gateway.nginx_proxy },
          { label: 'HTTPS / TLS', status: systemStatus.gateway.https_active ? 'active' : 'unhealthy' },
          { label: 'Domain', status: 'info', detail: systemStatus.gateway.domain },
        ],
      },
      {
        title: 'Data',
        items: [
          { label: systemStatus.data.postgres.label, status: systemStatus.data.postgres.status },
          { label: systemStatus.data.redis.label, status: systemStatus.data.redis.status },
          { label: systemStatus.data.qdrant.label, status: systemStatus.data.qdrant.status },
          { label: systemStatus.data.minio.label, status: systemStatus.data.minio.status },
        ],
      },
      {
        title: 'AI',
        items: [
          { label: 'Active Provider', status: 'info', detail: systemStatus.ai_provider.active_provider },
          { label: 'Model', status: 'info', detail: systemStatus.ai_provider.model },
          { label: 'LiteLLM Gateway', status: systemStatus.ai_provider.litellm_enabled ? 'active' : 'disabled', detail: systemStatus.ai_provider.litellm_enabled ? 'Enabled' : 'Disabled' },
        ],
      },
    ]
  }

  const groups = buildGroups()

  // KPI data
  const kpiData = systemStatus
    ? [
        { label: 'Overall Health', value: systemStatus.overall_healthy ? 'Healthy' : 'Degraded', tone: 'green' as const, icon: <span className="text-emerald-600 dark:text-emerald-400">{Icons.checkmark}</span> },
        { label: 'HTTPS / TLS', value: systemStatus.gateway.https_active ? 'Active' : 'Inactive', tone: 'blue' as const, icon: <span className="text-sky-600 dark:text-sky-400">{Icons.lock}</span> },
        { label: 'RAG Evaluation', value: `${systemStatus.rag_quality.passed_tests}/${systemStatus.rag_quality.total_tests} PASS`, tone: 'green' as const, icon: <span className="text-emerald-600 dark:text-emerald-400">{Icons.chart}</span> },
        { label: 'Active Provider', value: systemStatus.ai_provider.active_provider.charAt(0).toUpperCase() + systemStatus.ai_provider.active_provider.slice(1), tone: 'blue' as const, icon: <span className="text-sky-600 dark:text-sky-400">{Icons.lightning}</span> },
        { label: 'Documents Indexed', value: String(systemStatus.documents.indexed), tone: 'green' as const, icon: <span className="text-emerald-600 dark:text-emerald-400">{Icons.document}</span> },
      ]
    : null

  // -------------------------------------------------------------------------
  // Non-admin simplified dashboard
  // -------------------------------------------------------------------------
  if (!isAdmin) {
    return (
      <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg">
        <AppHeader />
        <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">

          {/* Hero */}
          <div className="card p-8 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-hiplink-blue/5 to-transparent pointer-events-none" />
            <div className="relative z-10">
              <Image src="/hiplink-logo.png" alt="HipLink" width={72} height={72} className="object-contain mx-auto mb-4" />
              <h1 className="text-2xl font-bold text-hiplink-dark dark:text-dark-text mb-2">Enterprise AI Assistant</h1>
              <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted max-w-xl mx-auto mb-4 leading-relaxed">
                Chat with AI using general conversation or query your uploaded documents with RAG-powered retrieval.
              </p>
              <div className="flex items-center justify-center gap-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">
                <span>Production • chatbot.hiplink.com</span>
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium ${statusBadgeClass('active')}`}>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  Online
                </span>
              </div>
            </div>
          </div>

          {/* Operations (non-admin view) */}
          <div>
            <h2 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text uppercase tracking-wider mb-4">Operations</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <OperationCard title="Chat" description="Start a conversation with the AI assistant." href="/chat" icon={Icons.chat} color="bg-hiplink-blue" />
              {perms.canViewDocuments && (
                <OperationCard title="Documents" description="Upload and manage your documents for RAG." href="/documents" icon={Icons.document} color="bg-emerald-500" />
              )}
            </div>
          </div>

          {/* Welcome footer */}
          <div className="card p-6 text-center">
            <p className="text-sm text-hiplink-dark dark:text-dark-text">
              Welcome back, <span className="font-semibold">{displayName}</span>
            </p>
            <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mt-1">
              Role: <span className="capitalize font-medium text-hiplink-dark dark:text-dark-text">{rawRole}</span>
            </p>
          </div>
        </div>
      </main>
    )
  }

  // -------------------------------------------------------------------------
  // Admin full dashboard
  // -------------------------------------------------------------------------
  return (
    <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg">
      <AppHeader />
      <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">

        {/* Hero */}
        <div className="card p-8 text-center relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-hiplink-blue/5 to-transparent pointer-events-none" />
          <div className="relative z-10">
            <Image src="/hiplink-logo.png" alt="HipLink" width={72} height={72} className="object-contain mx-auto mb-4" />
            <h1 className="text-2xl font-bold text-hiplink-dark dark:text-dark-text mb-2">Enterprise AI Assistant</h1>
            <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted max-w-xl mx-auto mb-4 leading-relaxed">
              Chat with AI using general conversation or query your uploaded documents with RAG-powered retrieval.
            </p>
            <div className="flex items-center justify-center gap-3 text-xs">
              <span className="text-hiplink-secondary dark:text-dark-text-dim">
                Production • {systemStatus?.gateway.domain ?? (typeof window !== 'undefined' ? window.location.hostname : 'chatbot.hiplink.com')}
              </span>
              {loading && !systemStatus ? (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-500/20">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                  Checking…
                </span>
              ) : statusError ? (
                <RefreshIndicator error={statusError} />
              ) : (
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium ${statusBadgeClass('active')}`}>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  Operational
                </span>
              )}
              {lastRefresh && (
                <span className="text-hiplink-secondary dark:text-dark-text-dim" title={lastRefresh.toLocaleString()}>
                  Updated {lastRefresh.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* KPI Row */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {kpiData ? (
            kpiData.map((k) => <KPICard key={k.label} {...k} />)
          ) : (
            Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="card p-4 space-y-2">
                <SkeletonRow w="w-1/2" />
                <SkeletonRow w="w-2/3" />
              </div>
            ))
          )}
        </div>

        {/* Service Health */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text uppercase tracking-wider">Service Health</h2>
            <RefreshIndicator error={statusError} />
          </div>
          {loading && !systemStatus ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {groups.map((group) => (
                <ServiceHealthGroup key={group.title} {...group} />
              ))}
            </div>
          )}
          {systemStatus && (
            <p className="text-[11px] text-hiplink-secondary dark:text-dark-text-dim mt-3 italic">
              {systemStatus.status_source_note}
            </p>
          )}
        </div>

        {/* Operations */}
        <div>
          <h2 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text uppercase tracking-wider mb-4">Operations</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <OperationCard title="Chat" description="General conversation with AI or RAG-powered knowledge base queries." href="/chat" icon={Icons.chat} color="bg-hiplink-blue" />
            {perms.canViewDocuments && (
              <OperationCard title="Documents" description="Upload, manage, and index documents for RAG retrieval." href="/documents" icon={Icons.document} color="bg-emerald-500" />
            )}
            <OperationCard title="Observability" description="Monitor usage, latency, feedback, and blocked answers." href="/admin/observability" icon={Icons.eye} color="bg-sky-500" />
            <OperationCard title="Evaluations" description="Track controlled RAG quality tests and failure analysis." href="/admin/evaluations" icon={Icons.chart} color="bg-purple-500" />
            <OperationCard title="Users" description="Manage user accounts, roles, and permissions." href="/admin/users" icon={Icons.users} color="bg-amber-500" />
            <OperationCard title="Audit Logs" description="Review security audit events and user actions." href="/admin/audit-logs" icon={Icons.shield} color="bg-rose-500" />
          </div>
        </div>

        {/* Bottom Row: RAG & Data | Security | Recent Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* RAG & Data Summary */}
          <div className="card p-5">
            <h3 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text mb-4 flex items-center gap-2">
              <span className="text-emerald-600 dark:text-emerald-400">{Icons.chart}</span>
              RAG &amp; Data Summary
            </h3>
            {!systemStatus ? (
              <div className="space-y-2">
                <SkeletonRow />
                <SkeletonRow w="w-2/3" />
              </div>
            ) : (
              <div className="space-y-2.5 text-sm">
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Total Uploaded</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium">{systemStatus.documents.total_documents}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Indexed</span>
                  <span className="text-emerald-600 dark:text-emerald-400 font-medium">{systemStatus.documents.indexed}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Failed</span>
                  <span className="text-red-600 dark:text-red-400 font-medium">{systemStatus.documents.failed}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Pending</span>
                  <span className="text-amber-600 dark:text-amber-400 font-medium">{systemStatus.documents.pending}</span>
                </div>
                <div className="h-px bg-hiplink-border dark:bg-dark-border my-2" />
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Collection</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium">{systemStatus.documents.collection_name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Embedding</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium">{systemStatus.documents.embedding_provider}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Dimension</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium">{systemStatus.documents.embedding_dimension}D</span>
                </div>
                <div className="h-px bg-hiplink-border dark:bg-dark-border my-2" />
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">RAG Pass Rate</span>
                  <span className="text-emerald-600 dark:text-emerald-400 font-medium">{systemStatus.rag_quality.pass_percentage}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Total Cases</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium">{systemStatus.rag_quality.total_tests}</span>
                </div>
                {systemStatus.rag_quality.last_run && (
                  <p className="text-[11px] text-hiplink-secondary dark:text-dark-text-dim pt-1">
                    Last run: {new Date(systemStatus.rag_quality.last_run).toLocaleString()}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Security & Admin Summary */}
          <div className="card p-5">
            <h3 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text mb-4 flex items-center gap-2">
              <span className="text-amber-600 dark:text-amber-400">{Icons.lock}</span>
              Security &amp; Admin Summary
            </h3>
            {!systemStatus ? (
              <div className="space-y-2">
                <SkeletonRow />
                <SkeletonRow w="w-2/3" />
              </div>
            ) : (
              <div className="space-y-2.5 text-sm">
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Auth Mode</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium capitalize">{systemStatus.security.auth_mode}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Current Role</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium capitalize">{systemStatus.security.current_user_role}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Active Users</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium">{systemStatus.security.total_active_users}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Failed Logins (24h)</span>
                  <span className={`font-medium ${systemStatus.security.recent_failed_logins_24h > 0 ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                    {systemStatus.security.recent_failed_logins_24h}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Audit Events (24h)</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium">{systemStatus.security.recent_audit_events_24h}</span>
                </div>
                <div className="h-px bg-hiplink-border dark:bg-dark-border my-2" />
                <div className="flex justify-between">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted">Last Admin Action</span>
                  <span className="text-hiplink-dark dark:text-dark-text font-medium truncate max-w-[120px]">
                    {systemStatus.security.last_admin_action
                      ? systemStatus.security.last_admin_action.action
                      : 'N/A'}
                  </span>
                </div>
                {systemStatus.security.last_admin_action?.timestamp && (
                  <p className="text-[11px] text-hiplink-secondary dark:text-dark-text-dim">
                    {new Date(systemStatus.security.last_admin_action.timestamp).toLocaleString()}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Recent Activity */}
          <div className="card p-5">
            <h3 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text mb-4 flex items-center gap-2">
              <span className="text-sky-600 dark:text-sky-400">{Icons.activity}</span>
              Recent Activity
            </h3>
            {!systemStatus ? (
              <div className="space-y-2">
                <SkeletonRow />
                <SkeletonRow w="w-2/3" />
                <SkeletonRow w="w-3/4" />
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <h4 className="text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim mb-2">Documents</h4>
                  {systemStatus.recent_activity.recent_documents.length > 0 ? (
                    <ul className="space-y-1.5">
                      {systemStatus.recent_activity.recent_documents.map((doc) => (
                        <li key={doc.id} className="flex items-center justify-between text-xs">
                          <span className="text-hiplink-dark dark:text-dark-text truncate max-w-[180px]">{doc.filename}</span>
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${statusBadgeClass(doc.status)}`}>
                            {doc.status}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim">No recent documents.</p>
                  )}
                </div>

                <div>
                  <h4 className="text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim mb-2">Audit Events</h4>
                  {systemStatus.recent_activity.recent_audit_events.length > 0 ? (
                    <ul className="space-y-1.5">
                      {systemStatus.recent_activity.recent_audit_events.map((evt) => (
                        <li key={evt.id} className="flex items-center justify-between text-xs">
                          <span className="text-hiplink-dark dark:text-dark-text truncate max-w-[140px]">{evt.action}</span>
                          <span className="text-hiplink-secondary dark:text-dark-text-dim">{evt.username}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim">No recent events.</p>
                  )}
                </div>

                {systemStatus.recent_activity.latest_evaluation && (
                  <div>
                    <h4 className="text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim mb-2">Latest Evaluation</h4>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-hiplink-dark dark:text-dark-text">{systemStatus.recent_activity.latest_evaluation.status}</span>
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">{systemStatus.recent_activity.latest_evaluation.pass_percentage}%</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-[11px] text-hiplink-secondary dark:text-dark-text-dim pt-4">
          Welcome back, <span className="text-hiplink-dark dark:text-dark-text font-medium">{displayName}</span>
          {lastRefresh && (
            <span className="ml-2" title={lastRefresh.toLocaleString()}>
              • Last updated {lastRefresh.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
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