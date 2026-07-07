'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import Link from 'next/link'
import HipLinkLogo from '@/components/HipLinkLogo'
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
// Status helpers
// ---------------------------------------------------------------------------
function statusDot(status: string) {
  const s = status.toLowerCase()
  if (['healthy', 'active', 'pass', 'indexed', 'success', 'operational', 'online'].includes(s)) return 'bg-emerald-500'
  if (['unhealthy', 'failure', 'failed', 'error', 'degraded', 'offline'].includes(s)) return 'bg-red-500'
  if (['warning', 'pending', 'unknown'].includes(s)) return 'bg-amber-500'
  return 'bg-sky-500'
}

function statusColor(status: string) {
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

// Badge pill variant for inline status
function badgeClass(status: string) {
  const s = status.toLowerCase()
  if (['healthy', 'active', 'pass', 'indexed', 'success', 'operational', 'online'].includes(s)) {
    return 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20'
  }
  if (['unhealthy', 'failure', 'failed', 'error', 'degraded', 'offline'].includes(s)) {
    return 'bg-red-50 dark:bg-red-500/10 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-500/20'
  }
  if (['warning', 'pending', 'unknown'].includes(s)) {
    return 'bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-500/20'
  }
  return 'bg-sky-50 dark:bg-sky-500/10 text-sky-700 dark:text-sky-400 border border-sky-200 dark:border-sky-500/20'
}

// Skeleton helpers
function SkeletonBar({ w = 'w-3/4', h = 'h-3' }: { w?: string; h?: string }) {
  return <div className={`${h} bg-gray-200 dark:bg-slate-700/50 rounded ${w} animate-pulse`} />
}

// Inline refresh indicator (small, non-intrusive)
function RefreshIndicator({ error }: { error: string | null }) {
  if (!error) return null
  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-500/20" title={error}>
      <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
      Refresh failed
    </span>
  )
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------
const iconCheck = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 13l4 4L19 7" /></svg>
)
const iconLock = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
)
const iconChart = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
)
const iconBolt = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
)
const iconDoc = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
)
const iconChat = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>
)
const iconEye = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
)
const iconUsers = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" /></svg>
)
const iconShield = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>
)
const iconActivity = (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
)
const iconClock = (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
)
const iconChevron = (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
)

// ---------------------------------------------------------------------------
// KPI Card — label / large value / subtext pattern
// ---------------------------------------------------------------------------
function KPICard({ label, value, subtext, icon, color }: { label: string; value: string; subtext: string; icon: React.ReactNode; color: string }) {
  return (
    <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-5 flex items-center gap-4 h-[96px]">
      <div className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 ${color}`}>
        <div className="text-white">{icon}</div>
      </div>
      <div className="min-w-0">
        <p className="text-sm font-medium text-hiplink-secondary dark:text-dark-text-muted leading-tight">{label}</p>
        <p className="text-2xl font-bold text-hiplink-dark dark:text-dark-text leading-tight mt-0.5">{value}</p>
        <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim leading-tight mt-0.5">{subtext}</p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Status row — with subtle background strip
// ---------------------------------------------------------------------------
function StatusRow({ label, status, detail }: ServiceItem) {
  return (
    <div className="flex items-center justify-between py-2.5 px-3 rounded-lg hover:bg-gray-50 dark:hover:bg-dark-elevated transition-colors">
      <span className="text-sm text-hiplink-dark dark:text-dark-text font-medium">{label}</span>
      <span className="flex items-center gap-2 flex-shrink-0 ml-3">
        <span className={`w-2 h-2 rounded-full ${statusDot(status)}`} />
        <span className={`text-sm font-semibold ${statusColor(status)}`}>
          {detail || status}
        </span>
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Unified Service Health Panel
// ---------------------------------------------------------------------------
function ServiceHealthPanel({ groups, error }: { groups: StatusGroup[]; error: string | null }) {
  return (
    <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-hiplink-border dark:border-dark-border flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-hiplink-dark dark:text-dark-text">Service Health</h2>
          <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted mt-0.5">System infrastructure and connectivity overview</p>
        </div>
        <RefreshIndicator error={error} />
      </div>

      {/* 4 columns */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 divide-hiplink-border dark:divide-dark-border">
        {groups.map((group, idx) => (
          <div
            key={group.title}
            className={`p-5 ${
              idx < groups.length - 1
                ? 'lg:border-r border-hiplink-border dark:border-dark-border'
                : ''
            } ${
              idx % 2 === 0
                ? 'sm:border-r sm:[&:nth-child(2)]:border-r-0 lg:[&:nth-child(2)]:border-r border-hiplink-border dark:border-dark-border'
                : ''
            }`}
          >
            <h3 className="text-sm font-bold text-hiplink-dark dark:text-dark-text mb-3">{group.title}</h3>
            <div className="space-y-1">
              {group.items.map((item) => (
                <StatusRow key={item.label} label={item.label} status={item.status} detail={item.detail} />
              ))}
            </div>
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
      className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-6 flex items-start gap-4 hover:shadow-xl hover:border-hiplink-blue/40 dark:hover:border-hiplink-blue/30 transition-all group h-full"
    >
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${color}`}>
        <div className="text-white">{icon}</div>
      </div>
      <div className="flex-1 min-w-0">
        <h3 className="text-base font-bold text-hiplink-dark dark:text-dark-text group-hover:text-hiplink-blue dark:group-hover:text-sky-400 transition-colors">
          {title}
        </h3>
        <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim mt-1 leading-relaxed">
          {description}
        </p>
      </div>
      <div className="text-hiplink-secondary dark:text-dark-text-dim flex-shrink-0 group-hover:text-hiplink-blue dark:group-hover:text-sky-400 group-hover:translate-x-0.5 transition-all">
        {iconChevron}
      </div>
    </Link>
  )
}

// ---------------------------------------------------------------------------
// Metric Tile — name + value side by side
// ---------------------------------------------------------------------------
function MetricTile({ label, value, tone }: { label: string; value: string; tone?: 'green' | 'red' | 'amber' | 'default' }) {
  const color = tone === 'green' ? 'text-emerald-600 dark:text-emerald-400'
    : tone === 'red' ? 'text-red-600 dark:text-red-400'
      : tone === 'amber' ? 'text-amber-600 dark:text-amber-400'
        : 'text-hiplink-dark dark:text-dark-text'
  return (
    <div>
      <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim">{label}</p>
      <p className={`text-lg font-bold ${color}`}>{value}</p>
    </div>
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

  const authFetchRef = useRef(authFetch)
  const abortRef = useRef<AbortController | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const fetchInProgressRef = useRef(false)

  authFetchRef.current = authFetch

  const loadStatus = useCallback(async (isBackground = false) => {
    if (!isAdmin) { setLoading(false); return }
    if (fetchInProgressRef.current) return

    fetchInProgressRef.current = true
    if (!isBackground) setLoading(true)
    setStatusError(null)

    try {
      if (abortRef.current) abortRef.current.abort()
      abortRef.current = new AbortController()
      const res = await authFetchRef.current('/api/admin/system/status', { signal: abortRef.current.signal })
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
      if (err?.name === 'AbortError') return
      setStatusError('Refresh failed')
    } finally {
      fetchInProgressRef.current = false
      if (!isBackground) setLoading(false)
    }
  }, [isAdmin])

  useEffect(() => {
    if (authLoading) return
    loadStatus(false)
    intervalRef.current = setInterval(() => loadStatus(true), 60000)
    return () => {
      if (abortRef.current) abortRef.current.abort()
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [authLoading, loadStatus])

  const displayName = user?.full_name || user?.username || user?.email || auth?.username || ''

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
      { title: 'Application', items: [
        { label: 'Frontend UI', status: systemStatus.application.frontend.status },
        { label: 'Backend API', status: systemStatus.application.backend_api.status },
      ]},
      { title: 'Gateway', items: [
        { label: 'Nginx Reverse Proxy', status: systemStatus.gateway.nginx_proxy },
        { label: 'HTTPS / TLS', status: systemStatus.gateway.https_active ? 'active' : 'unhealthy' },
        { label: 'Domain', status: 'info', detail: systemStatus.gateway.domain },
      ]},
      { title: 'Data', items: [
        { label: 'PostgreSQL DB', status: systemStatus.data.postgres.status },
        { label: 'Redis Cache', status: systemStatus.data.redis.status },
        { label: 'Qdrant Vector DB', status: systemStatus.data.qdrant.status },
        { label: 'MinIO Object Storage', status: systemStatus.data.minio.status },
      ]},
      { title: 'AI', items: [
        { label: 'Active Provider', status: 'info', detail: systemStatus.ai_provider.active_provider },
        { label: 'Model', status: 'info', detail: systemStatus.ai_provider.model },
        { label: 'LiteLLM Gateway', status: systemStatus.ai_provider.litellm_enabled ? 'active' : 'disabled', detail: systemStatus.ai_provider.litellm_enabled ? 'Enabled' : 'Disabled' },
      ]},
    ]
  }

  const groups = buildGroups()

  const kpiData = systemStatus
    ? ([
        { label: 'Overall Health', value: systemStatus.overall_healthy ? 'Healthy' : 'Degraded', subtext: 'All systems operational', icon: iconCheck, color: 'bg-emerald-500' },
        { label: 'HTTPS / TLS', value: systemStatus.gateway.https_active ? 'Active' : 'Inactive', subtext: 'Secure connection enabled', icon: iconLock, color: 'bg-sky-500' },
        { label: 'RAG Evaluation', value: `${systemStatus.rag_quality.passed_tests}/${systemStatus.rag_quality.total_tests} PASS`, subtext: 'Latest evaluation result', icon: iconChart, color: 'bg-emerald-500' },
        { label: 'Active Provider', value: systemStatus.ai_provider.active_provider.charAt(0).toUpperCase() + systemStatus.ai_provider.active_provider.slice(1), subtext: 'LLM request routing', icon: iconBolt, color: 'bg-sky-500' },
        { label: 'Documents Indexed', value: String(systemStatus.documents.indexed), subtext: `Out of ${systemStatus.documents.total_documents} uploaded`, icon: iconDoc, color: 'bg-emerald-500' },
      ])
    : null

  // ========================================================================
  // NON-ADMIN DASHBOARD
  // ========================================================================
  if (!isAdmin) {
    return (
      <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg">
        <AppHeader />
        <div className="max-w-[1440px] mx-auto px-6 py-10 space-y-8">

          {/* Hero */}
          <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-2xl py-16 px-8 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-hiplink-blue/5 to-transparent pointer-events-none" />
            <div className="relative z-10">
              <HipLinkLogo variant="auto" size="hero" priority className="mx-auto mb-6" />
              <h1 className="text-4xl font-bold text-hiplink-dark dark:text-dark-text mb-3">Enterprise AI Assistant</h1>
              <p className="text-base text-hiplink-secondary dark:text-dark-text-muted max-w-lg mx-auto mb-6 leading-relaxed">
                Chat with AI using general conversation or query your uploaded documents with RAG-powered retrieval.
              </p>
              <div className="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-gray-50 dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border">
                <span className="text-sm text-hiplink-secondary dark:text-dark-text-dim">Production • chatbot.hiplink.com</span>
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-sm font-medium ${badgeClass('active')}`}>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  Online
                </span>
              </div>
            </div>
          </div>

          {/* Operations */}
          <div>
            <h2 className="text-lg font-bold text-hiplink-dark dark:text-dark-text mb-5">Operations</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
              <OperationCard title="Chat" description="Start a conversation with the AI assistant." href="/chat" icon={iconChat} color="bg-hiplink-blue" />
              {perms.canViewDocuments && (
                <OperationCard title="Documents" description="Upload and manage your documents for RAG." href="/documents" icon={iconDoc} color="bg-emerald-500" />
              )}
            </div>
          </div>

          {/* Welcome */}
          <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-8 text-center">
            <p className="text-base text-hiplink-dark dark:text-dark-text">
              Welcome back, <span className="font-bold">{displayName}</span>
            </p>
            <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim mt-1">
              Role: <span className="capitalize font-semibold text-hiplink-dark dark:text-dark-text">{rawRole}</span>
            </p>
          </div>
        </div>
      </main>
    )
  }

  // ========================================================================
  // ADMIN FULL DASHBOARD
  // ========================================================================
  return (
    <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg">
      <AppHeader />
      <div className="max-w-[1440px] mx-auto px-6 py-10 space-y-8">

        {/* Hero */}
        <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-2xl py-16 px-8 text-center relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-hiplink-blue/5 via-transparent to-transparent pointer-events-none" />
          <div className="relative z-10">
            <HipLinkLogo variant="auto" size="hero" priority className="mx-auto mb-6" />
            <h1 className="text-4xl font-bold text-hiplink-dark dark:text-dark-text mb-3">Enterprise AI Assistant</h1>
            <p className="text-base text-hiplink-secondary dark:text-dark-text-muted max-w-lg mx-auto mb-6 leading-relaxed">
              Chat with AI using general conversation or query your uploaded documents with RAG-powered retrieval.
            </p>
            <div className="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-gray-50 dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border">
              <span className="text-sm text-hiplink-secondary dark:text-dark-text-dim">
                Production • {systemStatus?.gateway.domain ?? (typeof window !== 'undefined' ? window.location.hostname : 'chatbot.hiplink.com')}
              </span>
              <span className="w-px h-3 bg-hiplink-border dark:bg-dark-border" />
              {loading && !systemStatus ? (
                <span className="inline-flex items-center gap-1.5 text-sm font-medium text-amber-600 dark:text-amber-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                  Checking…
                </span>
              ) : statusError ? (
                <RefreshIndicator error={statusError} />
              ) : (
                <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${badgeClass('active')}`}>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  Operational
                </span>
              )}
              {lastRefresh && (
                <>
                  <span className="w-px h-3 bg-hiplink-border dark:bg-dark-border" />
                  <span className="text-xs text-hiplink-secondary dark:text-dark-text-dim flex items-center gap-1" title={lastRefresh.toLocaleString()}>
                    {iconClock}
                    {lastRefresh.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* KPI Row */}
        <div>
          <h2 className="text-lg font-bold text-hiplink-dark dark:text-dark-text mb-5">Key Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-5">
            {kpiData ? (
              kpiData.map((k) => <KPICard key={k.label} {...k} />)
            ) : (
              Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-5 space-y-2 h-[96px]">
                  <SkeletonBar w="w-1/2" />
                  <SkeletonBar w="w-3/4" h="h-6" />
                  <SkeletonBar w="w-2/3" />
                </div>
              ))
            )}
          </div>
        </div>

        {/* Service Health */}
        {loading && !systemStatus ? (
          <div className="space-y-4">
            <SkeletonBar w="w-48" h="h-6" />
            <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl overflow-hidden">
              <div className="p-6 space-y-3">
                <SkeletonBar w="w-1/3" />
                <SkeletonBar w="w-full" />
                <SkeletonBar w="w-2/3" />
              </div>
            </div>
          </div>
        ) : (
          <ServiceHealthPanel groups={groups} error={statusError} />
        )}
        {systemStatus && (
          <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim italic">
            {systemStatus.status_source_note}
          </p>
        )}

        {/* Operations */}
        <div>
          <h2 className="text-lg font-bold text-hiplink-dark dark:text-dark-text mb-5">Operations</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            <OperationCard title="Chat" description="General conversation with AI or RAG-powered knowledge base queries." href="/chat" icon={iconChat} color="bg-hiplink-blue" />
            {perms.canViewDocuments && (
              <OperationCard title="Documents" description="Upload, manage, and index documents for RAG retrieval." href="/documents" icon={iconDoc} color="bg-emerald-500" />
            )}
            <OperationCard title="Observability" description="Monitor usage, latency, feedback, and blocked answers." href="/admin/observability" icon={iconEye} color="bg-sky-500" />
            <OperationCard title="Evaluations" description="Track controlled RAG quality tests and failure analysis." href="/admin/evaluations" icon={iconChart} color="bg-purple-500" />
            <OperationCard title="Users" description="Manage user accounts, roles, and permissions." href="/admin/users" icon={iconUsers} color="bg-amber-500" />
            <OperationCard title="Audit Logs" description="Review security audit events and user actions." href="/admin/audit-logs" icon={iconShield} color="bg-rose-500" />
          </div>
        </div>

        {/* Bottom Row: RAG & Data | Security | Recent Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* RAG & Data Summary */}
          <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl overflow-hidden flex flex-col">
            <div className="px-6 py-4 border-b border-hiplink-border dark:border-dark-border flex items-center gap-2">
              <span className="text-emerald-600 dark:text-emerald-400">{iconChart}</span>
              <h3 className="text-base font-bold text-hiplink-dark dark:text-dark-text">RAG &amp; Data Summary</h3>
            </div>
            <div className="p-6 flex-1">
              {!systemStatus ? (
                <div className="space-y-3">
                  <SkeletonBar w="w-3/4" />
                  <SkeletonBar w="w-1/2" />
                </div>
              ) : (
                <div className="space-y-5">
                  {/* Document metrics tiles */}
                  <div className="grid grid-cols-2 gap-4">
                    <MetricTile label="Total Uploaded" value={String(systemStatus.documents.total_documents)} />
                    <MetricTile label="Indexed" value={String(systemStatus.documents.indexed)} tone="green" />
                    <MetricTile label="Failed" value={String(systemStatus.documents.failed)} tone="red" />
                    <MetricTile label="Pending" value={String(systemStatus.documents.pending)} tone="amber" />
                  </div>

                  <div className="h-px bg-hiplink-border dark:bg-dark-border" />

                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-hiplink-secondary dark:text-dark-text-muted">Collection</span>
                      <span className="text-hiplink-dark dark:text-dark-text font-semibold">{systemStatus.documents.collection_name}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-hiplink-secondary dark:text-dark-text-muted">Embedding</span>
                      <span className="text-hiplink-dark dark:text-dark-text font-semibold">{systemStatus.documents.embedding_provider}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-hiplink-secondary dark:text-dark-text-muted">Dimension</span>
                      <span className="text-hiplink-dark dark:text-dark-text font-semibold">{systemStatus.documents.embedding_dimension}D</span>
                    </div>
                  </div>

                  <div className="h-px bg-hiplink-border dark:bg-dark-border" />

                  <div className="grid grid-cols-2 gap-4">
                    <MetricTile label="RAG Pass Rate" value={`${systemStatus.rag_quality.pass_percentage}%`} tone="green" />
                    <MetricTile label="Total Cases" value={String(systemStatus.rag_quality.total_tests)} />
                  </div>

                  {systemStatus.rag_quality.last_run && (
                    <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">
                      Last run: {new Date(systemStatus.rag_quality.last_run).toLocaleString()}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Security & Admin Summary */}
          <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl overflow-hidden flex flex-col">
            <div className="px-6 py-4 border-b border-hiplink-border dark:border-dark-border flex items-center gap-2">
              <span className="text-amber-600 dark:text-amber-400">{iconLock}</span>
              <h3 className="text-base font-bold text-hiplink-dark dark:text-dark-text">Security &amp; Admin Summary</h3>
            </div>
            <div className="p-6 flex-1">
              {!systemStatus ? (
                <div className="space-y-3">
                  <SkeletonBar w="w-3/4" />
                  <SkeletonBar w="w-1/2" />
                </div>
              ) : (
                <div className="space-y-5">
                  <div className="grid grid-cols-2 gap-4">
                    <MetricTile label="Failed Logins" value={String(systemStatus.security.recent_failed_logins_24h)} tone={systemStatus.security.recent_failed_logins_24h > 0 ? 'red' : 'green'} />
                    <MetricTile label="Audit Events" value={String(systemStatus.security.recent_audit_events_24h)} />
                    <MetricTile label="Active Users" value={String(systemStatus.security.total_active_users)} />
                    <MetricTile label="Auth Mode" value={systemStatus.security.auth_mode} />
                  </div>

                  <div className="h-px bg-hiplink-border dark:bg-dark-border" />

                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between items-start gap-3">
                      <span className="text-hiplink-secondary dark:text-dark-text-muted">Last Admin Action</span>
                      <span className="text-hiplink-dark dark:text-dark-text font-semibold text-right truncate max-w-[160px]">
                        {systemStatus.security.last_admin_action ? systemStatus.security.last_admin_action.action : 'N/A'}
                      </span>
                    </div>
                    {systemStatus.security.last_admin_action?.timestamp && (
                      <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim text-right">
                        {new Date(systemStatus.security.last_admin_action.timestamp).toLocaleString()}
                      </p>
                    )}
                    <div className="pt-1">
                      <Link href="/admin/security" className="inline-flex items-center gap-1.5 text-sm font-medium text-hiplink-blue hover:underline">
                        Open Security Operations
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
                      </Link>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl overflow-hidden flex flex-col">
            <div className="px-6 py-4 border-b border-hiplink-border dark:border-dark-border flex items-center gap-2">
              <span className="text-sky-600 dark:text-sky-400">{iconActivity}</span>
              <h3 className="text-base font-bold text-hiplink-dark dark:text-dark-text">Recent Activity</h3>
            </div>
            <div className="p-6 flex-1">
              {!systemStatus ? (
                <div className="space-y-3">
                  <SkeletonBar w="w-3/4" />
                  <SkeletonBar w="w-1/2" />
                  <SkeletonBar w="w-2/3" />
                </div>
              ) : (
                <div className="space-y-6 text-sm">
                  {/* Documents */}
                  <div>
                    <h4 className="text-sm font-bold text-hiplink-dark dark:text-dark-text mb-3">Recent Documents</h4>
                    {systemStatus.recent_activity.recent_documents.length > 0 ? (
                      <ul className="space-y-2">
                        {systemStatus.recent_activity.recent_documents.map((doc) => (
                          <li key={doc.id} className="flex items-center justify-between py-1.5">
                            <span className="text-sm text-hiplink-dark dark:text-dark-text truncate max-w-[200px]">{doc.filename}</span>
                            <span className={`px-2 py-0.5 rounded text-xs font-medium border ${badgeClass(doc.status)}`}>
                              {doc.status}
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No recent documents.</p>
                    )}
                  </div>

                  {/* Audit Events */}
                  <div>
                    <h4 className="text-sm font-bold text-hiplink-dark dark:text-dark-text mb-3">Recent Audit Events</h4>
                    {systemStatus.recent_activity.recent_audit_events.length > 0 ? (
                      <ul className="space-y-2">
                        {systemStatus.recent_activity.recent_audit_events.map((evt) => (
                          <li key={evt.id} className="flex items-center justify-between py-1.5">
                            <span className="text-sm text-hiplink-dark dark:text-dark-text truncate max-w-[160px]">{evt.action}</span>
                            <span className="text-sm text-hiplink-secondary dark:text-dark-text-dim">{evt.username}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No recent events.</p>
                    )}
                  </div>

                  {/* Latest Evaluation */}
                  {systemStatus.recent_activity.latest_evaluation && (
                    <div>
                      <h4 className="text-sm font-bold text-hiplink-dark dark:text-dark-text mb-3">Latest Evaluation</h4>
                      <div className="flex items-center justify-between py-1.5">
                        <span className="text-sm text-hiplink-dark dark:text-dark-text">{systemStatus.recent_activity.latest_evaluation.status}</span>
                        <span className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{systemStatus.recent_activity.latest_evaluation.pass_percentage}%</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-hiplink-secondary dark:text-dark-text-dim py-4">
          Welcome back, <span className="text-hiplink-dark dark:text-dark-text font-bold">{displayName}</span>
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