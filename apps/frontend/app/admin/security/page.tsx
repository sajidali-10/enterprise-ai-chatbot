'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { useAuthFetch } from '@/hooks/useApi'
import ProtectedRoute from '@/components/ProtectedRoute'
import AppHeader from '@/components/AppHeader'
import { normalizePermissions, type PermissionFlags } from '@/lib/permissions'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface FailedLoginEntry {
  id: number
  timestamp: string
  username: string
  request_ip: string | null
  user_agent: string
  status: string
  error_message: string | null
}

interface IPLoginCount {
  request_ip: string
  count: number
}

interface AdminActionEntry {
  id: number
  timestamp: string
  admin_username: string
  action: string
  target: string
  status: string
}

interface DocumentSecurityEntry {
  id: number
  timestamp: string
  username: string
  action: string
  document_name: string
  status: string
}

interface RAGSafetyEntry {
  id: number
  timestamp: string
  username: string
  action: string
  question_preview: string
  status: string
  error_message: string | null
  request_ip: string | null
}

interface RepeatedIP {
  ip: string
  failed_attempts: number
}

interface RepeatedUser {
  username: string
  denied_count: number
}

interface SummaryCards {
  failed_logins_24h: number
  successful_logins_24h: number
  admin_actions_24h: number
  permission_denied_24h: number
  document_access_changes_24h: number
  rag_fallbacks_24h: number
}

interface PasswordPolicy {
  minimum_length: number
  complexity_enabled: boolean
  weak_password_blocking_enabled: boolean
  password_reset_support: boolean
}

interface SessionManagement {
  auth_mode: string
  token_version_invalidation: boolean
  active_users_count: string
  force_logout_all_supported: boolean
  force_logout_all_status: string
}

interface LoginProtection {
  failed_login_audit_tracking: string
  repeated_ip_detection: string
  account_lockout: string
  failed_login_count_window: number
}

interface AuditEvidence {
  csv_export: string
  json_export: string
  export_limit: number
  audit_logs_link_available: boolean
}

interface RAGSafety {
  crag_fallback_checks: string
  high_risk_categories: string[]
  rag_evaluation_status: string
  unsupported_question_fallback: string
}

interface PolicyControls {
  password_policy: PasswordPolicy
  session_management: SessionManagement
  login_protection: LoginProtection
  audit_evidence: AuditEvidence
  rag_safety: RAGSafety
}

interface SecurityOverview {
  summary_cards: SummaryCards
  failed_login_activity: {
    events: FailedLoginEntry[]
    login_counts_by_ip: IPLoginCount[]
  }
  admin_actions: AdminActionEntry[]
  document_security: DocumentSecurityEntry[]
  rag_safety: RAGSafetyEntry[]
  risky_activity: {
    repeated_failed_login_ips: RepeatedIP[]
    repeated_permission_denied_users: RepeatedUser[]
  }
  export_options: {
    csv_supported: boolean
    json_supported: boolean
    query_params: string[]
  }
  query_window_hours: number
  queried_since: string
  policy_controls: PolicyControls
}

type TimeRange = 24 | 168 | 720

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function statusColor(status: string) {
  const s = status.toLowerCase()
  if (['success', 'healthy', 'pass', 'indexed', 'online'].includes(s)) return 'text-emerald-600 dark:text-emerald-400'
  if (['failure', 'failed', 'error', 'degraded', 'offline', 'unhealthy'].includes(s)) return 'text-red-600 dark:text-red-400'
  if (['warning', 'pending', 'unknown'].includes(s)) return 'text-amber-600 dark:text-amber-400'
  if (['fallback'].includes(s)) return 'text-purple-600 dark:text-purple-400'
  if (['blocked', 'high_risk'].includes(s)) return 'text-rose-600 dark:text-rose-400'
  return 'text-hiplink-dark dark:text-dark-text'
}

function badgeClass(status: string) {
  const s = status.toLowerCase()
  if (['success', 'healthy', 'pass', 'indexed'].includes(s)) {
    return 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20'
  }
  if (['failure', 'failed', 'error', 'degraded'].includes(s)) {
    return 'bg-red-50 dark:bg-red-500/10 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-500/20'
  }
  if (['warning', 'pending', 'unknown'].includes(s)) {
    return 'bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-500/20'
  }
  if (['fallback'].includes(s)) {
    return 'bg-purple-50 dark:bg-purple-500/10 text-purple-700 dark:text-purple-400 border border-purple-200 dark:border-purple-500/20'
  }
  if (['blocked', 'high_risk'].includes(s)) {
    return 'bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border border-rose-200 dark:border-rose-500/20'
  }
  return 'bg-sky-50 dark:bg-sky-500/10 text-sky-700 dark:text-sky-400 border border-sky-200 dark:border-sky-500/20'
}

function dotClass(status: string) {
  const s = status.toLowerCase()
  if (['success', 'healthy', 'pass', 'indexed', 'online'].includes(s)) return 'bg-emerald-500'
  if (['failure', 'failed', 'error', 'degraded', 'offline', 'unhealthy'].includes(s)) return 'bg-red-500'
  if (['warning', 'pending', 'unknown'].includes(s)) return 'bg-amber-500'
  if (['fallback', 'blocked', 'high_risk'].includes(s)) return 'bg-purple-500'
  return 'bg-sky-500'
}

function formatDate(iso: string | null) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function SkeletonBar({ w = 'w-3/4', h = 'h-3' }: { w?: string; h?: string }) {
  return <div className={`${h} bg-gray-200 dark:bg-slate-700/50 rounded ${w} animate-pulse`} />
}

// ---------------------------------------------------------------------------
// KPI Card
// ---------------------------------------------------------------------------
function KPICard({ label, value, color, icon }: { label: string; value: number | string; color: string; icon: React.ReactNode }) {
  return (
    <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-5 flex items-center gap-4 h-[96px]">
      <div className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 ${color}`}>
        <div className="text-white">{icon}</div>
      </div>
      <div className="min-w-0">
        <p className="text-sm font-medium text-hiplink-secondary dark:text-dark-text-muted leading-tight">{label}</p>
        <p className="text-2xl font-bold text-hiplink-dark dark:text-dark-text leading-tight mt-0.5">{value}</p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section wrapper
// ---------------------------------------------------------------------------
function Section({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl overflow-hidden">
      <div className="px-6 py-4 border-b border-hiplink-border dark:border-dark-border">
        <h2 className="text-base font-bold text-hiplink-dark dark:text-dark-text">{title}</h2>
        {subtitle && <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted mt-0.5">{subtitle}</p>}
      </div>
      <div className="p-6">
        {children}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------
const iconShield = <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>
const iconKey = <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" /></svg>
const iconUsers = <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" /></svg>
const iconLock = <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
const iconDoc = <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
const iconAlert = <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
const iconDownload = <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
const iconClock = <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
const iconRefresh = <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>

// ---------------------------------------------------------------------------
// Security page component
// ---------------------------------------------------------------------------
function SecurityPage() {
  const { auth } = useAuth()
  const authFetch = useAuthFetch()
  const rawRole = (auth?.role ?? 'viewer') as string
  const perms: PermissionFlags = normalizePermissions(auth?.permissions, rawRole)
  const isAdmin = perms.canAccessObservability

  const [data, setData] = useState<SecurityOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [hours, setHours] = useState<TimeRange>(24)

  const abortRef = useRef<AbortController | null>(null)

  const load = useCallback(async () => {
    if (!isAdmin) { setLoading(false); return }
    setLoading(true)
    setError(null)
    try {
      if (abortRef.current) abortRef.current.abort()
      abortRef.current = new AbortController()
      const res = await authFetch(`/api/admin/security/overview?hours=${hours}`, { signal: abortRef.current.signal })
      if (res.ok) {
        const json: SecurityOverview = await res.json()
        setData(json)
        setError(null)
      } else if (res.status === 401 || res.status === 403) {
        setError('Access denied. Admin privileges required.')
      } else {
        setError(`Failed to load security overview (HTTP ${res.status})`)
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') return
      setError('Failed to load security overview. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [isAdmin, hours, authFetch])

  useEffect(() => {
    load()
    return () => {
      if (abortRef.current) abortRef.current.abort()
    }
  }, [load])

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      const res = await authFetch(`/api/admin/security/export.${format}?hours=${hours}`)
      if (res.ok) {
        const blob = await res.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `audit-export.${format}`
        a.click()
        window.URL.revokeObjectURL(url)
      } else {
        setError(`Export failed (HTTP ${res.status})`)
      }
    } catch {
      setError('Export failed. Please try again.')
    }
  }

  if (!isAdmin) {
    return (
      <ProtectedRoute>
        <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg">
          <AppHeader />
          <div className="max-w-[1440px] mx-auto px-6 py-16 text-center">
            <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-8 max-w-md mx-auto">
              <h1 className="text-xl font-bold text-hiplink-dark dark:text-dark-text mb-2">Access Denied</h1>
              <p className="text-hiplink-secondary dark:text-dark-text-dim">Admin access is required to view Security Operations.</p>
            </div>
          </div>
        </main>
      </ProtectedRoute>
    )
  }

  return (
    <ProtectedRoute requirePermission="canAccessObservability">
      <main className="min-h-screen bg-hiplink-background dark:bg-dark-bg">
        <div className="max-w-[1440px] mx-auto px-6 py-8 space-y-8">

          {/* Hero / Header */}
          <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-2xl px-8 py-10 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-transparent pointer-events-none" />
            <div className="relative z-10">
              <h1 className="text-3xl font-bold text-hiplink-dark dark:text-dark-text mb-2">Security Operations</h1>
              <p className="text-base text-hiplink-secondary dark:text-dark-text-muted max-w-2xl mx-auto mb-6 leading-relaxed">
                Monitor authentication events, admin activity, document access changes, and RAG safety events.
              </p>
              <div className="flex items-center justify-center gap-4 flex-wrap">
                {/* Time range selector */}
                <div className="inline-flex rounded-lg border border-hiplink-border dark:border-dark-border bg-gray-50 dark:bg-dark-elevated overflow-hidden">
                  {([24, 168, 720] as TimeRange[]).map((h) => (
                    <button
                      key={h}
                      onClick={() => setHours(h)}
                      className={`px-4 py-2 text-sm font-medium transition-colors ${
                        hours === h
                          ? 'bg-hiplink-blue text-white'
                          : 'text-hiplink-secondary dark:text-dark-text-muted hover:bg-gray-100 dark:hover:bg-dark-border'
                      }`}
                    >
                      {h === 24 ? '24h' : h === 168 ? '7d' : '30d'}
                    </button>
                  ))}
                </div>

                <button
                  onClick={load}
                  disabled={loading}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-hiplink-blue text-white text-sm font-medium hover:bg-hiplink-blue/90 transition-colors disabled:opacity-50"
                >
                  {iconRefresh}
                  {loading ? 'Loading…' : 'Refresh'}
                </button>

                {data && (
                  <span className="text-xs text-hiplink-secondary dark:text-dark-text-dim flex items-center gap-1">
                    {iconClock}
                    {formatDate(data.queried_since)}
                  </span>
                )}
              </div>

              {error && (
                <div className="mt-4 px-4 py-2 rounded-lg bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20 text-amber-700 dark:text-amber-400 text-sm inline-block">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* KPI Cards */}
          <div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {loading ? (
                Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-5 h-[96px] space-y-2">
                    <SkeletonBar w="w-1/2" />
                    <SkeletonBar w="w-1/3" h="h-8" />
                  </div>
                ))
              ) : data ? (
                <>
                  <KPICard label="Failed Logins" value={data.summary_cards.failed_logins_24h} color="bg-red-500" icon={iconShield} />
                  <KPICard label="Successful Logins" value={data.summary_cards.successful_logins_24h} color="bg-emerald-500" icon={iconKey} />
                  <KPICard label="Admin Actions" value={data.summary_cards.admin_actions_24h} color="bg-sky-500" icon={iconUsers} />
                  <KPICard label="Permission Denied" value={data.summary_cards.permission_denied_24h} color="bg-amber-500" icon={iconLock} />
                  <KPICard label="Doc Access Changes" value={data.summary_cards.document_access_changes_24h} color="bg-amber-500" icon={iconDoc} />
                  <KPICard label="RAG Fallbacks" value={data.summary_cards.rag_fallbacks_24h} color="bg-purple-500" icon={iconAlert} />
                </>
              ) : null}
            </div>
          </div>

          {/* Failed Login Activity */}
          <Section title="Failed Login Activity" subtitle="Recent authentication failures and repeated attempts by IP">
            {loading ? (
              <div className="space-y-3">
                <SkeletonBar w="w-full" />
                <SkeletonBar w="w-full" />
                <SkeletonBar w="w-2/3" />
              </div>
            ) : !data || data.failed_login_activity.events.length === 0 ? (
              <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No failed login events in the selected time window.</p>
            ) : (
              <div className="space-y-6">
                {/* Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left border-b border-hiplink-border dark:border-dark-border text-hiplink-secondary dark:text-dark-text-dim">
                        <th className="pb-2 font-semibold">Time</th>
                        <th className="pb-2 font-semibold">Username</th>
                        <th className="pb-2 font-semibold">IP</th>
                        <th className="pb-2 font-semibold">User Agent</th>
                        <th className="pb-2 font-semibold">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                      {data.failed_login_activity.events.map((entry) => (
                        <tr key={entry.id} className="group">
                          <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted whitespace-nowrap">{formatDate(entry.timestamp)}</td>
                          <td className="py-2.5 text-hiplink-dark dark:text-dark-text font-medium">{entry.username}</td>
                          <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted font-mono text-xs">{entry.request_ip || '—'}</td>
                          <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted max-w-[200px] truncate" title={entry.user_agent}>{entry.user_agent || '—'}</td>
                          <td className="py-2.5">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium border ${badgeClass(entry.status)}`}>
                              {entry.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* IP counts */}
                {data.failed_login_activity.login_counts_by_ip.length > 0 && (
                  <div>
                    <h4 className="text-sm font-bold text-hiplink-dark dark:text-dark-text mb-3">Failed Logins by IP</h4>
                    <div className="flex flex-wrap gap-3">
                      {data.failed_login_activity.login_counts_by_ip.map((item) => (
                        <div key={item.request_ip} className="px-3 py-2 rounded-lg bg-gray-50 dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border">
                          <span className="text-xs text-hiplink-secondary dark:text-dark-text-muted">{item.request_ip}</span>
                          <span className="ml-2 text-sm font-bold text-red-600 dark:text-red-400">{item.count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </Section>

          {/* Admin Actions */}
          <Section title="Admin Actions" subtitle="Recent user management, role changes, and password resets">
            {loading ? (
              <div className="space-y-3">
                <SkeletonBar w="w-full" />
                <SkeletonBar w="w-full" />
              </div>
            ) : !data || data.admin_actions.length === 0 ? (
              <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No admin actions in the selected time window.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b border-hiplink-border dark:border-dark-border text-hiplink-secondary dark:text-dark-text-dim">
                      <th className="pb-2 font-semibold">Time</th>
                      <th className="pb-2 font-semibold">Admin</th>
                      <th className="pb-2 font-semibold">Action</th>
                      <th className="pb-2 font-semibold">Target</th>
                      <th className="pb-2 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                    {data.admin_actions.map((entry) => (
                      <tr key={entry.id}>
                        <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted whitespace-nowrap">{formatDate(entry.timestamp)}</td>
                        <td className="py-2.5 text-hiplink-dark dark:text-dark-text font-medium">{entry.admin_username}</td>
                        <td className="py-2.5 text-hiplink-dark dark:text-dark-text">{entry.action}</td>
                        <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted">{entry.target || '—'}</td>
                        <td className="py-2.5">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium border ${badgeClass(entry.status)}`}>
                            {entry.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          {/* Document Security */}
          <Section title="Document Security" subtitle="Uploads, deletions, and access changes">
            {loading ? (
              <div className="space-y-3">
                <SkeletonBar w="w-full" />
                <SkeletonBar w="w-full" />
              </div>
            ) : !data || data.document_security.length === 0 ? (
              <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No document security events in the selected time window.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b border-hiplink-border dark:border-dark-border text-hiplink-secondary dark:text-dark-text-dim">
                      <th className="pb-2 font-semibold">Time</th>
                      <th className="pb-2 font-semibold">User</th>
                      <th className="pb-2 font-semibold">Action</th>
                      <th className="pb-2 font-semibold">Document</th>
                      <th className="pb-2 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                    {data.document_security.map((entry) => (
                      <tr key={entry.id}>
                        <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted whitespace-nowrap">{formatDate(entry.timestamp)}</td>
                        <td className="py-2.5 text-hiplink-dark dark:text-dark-text font-medium">{entry.username}</td>
                        <td className="py-2.5 text-hiplink-dark dark:text-dark-text">{entry.action}</td>
                        <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted">{entry.document_name || '—'}</td>
                        <td className="py-2.5">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium border ${badgeClass(entry.status)}`}>
                            {entry.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          {/* RAG Safety Events */}
          <Section title="RAG Safety Events" subtitle="Fallback triggers, blocked questions, and low-confidence retrieval">
            {loading ? (
              <div className="space-y-3">
                <SkeletonBar w="w-full" />
                <SkeletonBar w="w-full" />
              </div>
            ) : !data || data.rag_safety.length === 0 ? (
              <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No RAG safety events in the selected time window.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b border-hiplink-border dark:border-dark-border text-hiplink-secondary dark:text-dark-text-dim">
                      <th className="pb-2 font-semibold">Time</th>
                      <th className="pb-2 font-semibold">User</th>
                      <th className="pb-2 font-semibold">Action</th>
                      <th className="pb-2 font-semibold">Question Preview</th>
                      <th className="pb-2 font-semibold">Status</th>
                      <th className="pb-2 font-semibold">IP</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                    {data.rag_safety.map((entry) => (
                      <tr key={entry.id}>
                        <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted whitespace-nowrap">{formatDate(entry.timestamp)}</td>
                        <td className="py-2.5 text-hiplink-dark dark:text-dark-text font-medium">{entry.username}</td>
                        <td className="py-2.5 text-hiplink-dark dark:text-dark-text">{entry.action}</td>
                        <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted max-w-[240px] truncate" title={entry.question_preview}>{entry.question_preview || '—'}</td>
                        <td className="py-2.5">
                          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium border ${badgeClass(entry.status)}`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${dotClass(entry.status)}`} />
                            {entry.status}
                          </span>
                        </td>
                        <td className="py-2.5 text-hiplink-secondary dark:text-dark-text-muted font-mono text-xs">{entry.request_ip || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          {/* Risky Activity */}
          <Section title="Risky Activity" subtitle="Repeated failed attempts and suspicious access patterns">
            {loading ? (
              <div className="space-y-3">
                <SkeletonBar w="w-full" />
                <SkeletonBar w="w-2/3" />
              </div>
            ) : !data ? (
              <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No data available.</p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Repeated IPs */}
                <div>
                  <h4 className="text-sm font-bold text-hiplink-dark dark:text-dark-text mb-3">Repeated Failed Login IPs</h4>
                  {data.risky_activity.repeated_failed_login_ips.length > 0 ? (
                    <div className="space-y-2">
                      {data.risky_activity.repeated_failed_login_ips.map((item) => (
                        <div key={item.ip} className="flex items-center justify-between py-2 px-3 rounded-lg bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/20">
                          <span className="text-sm text-hiplink-dark dark:text-dark-text font-mono">{item.ip}</span>
                          <span className="text-sm font-bold text-red-600 dark:text-red-400">{item.failed_attempts} failed</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No repeated failed login IPs detected.</p>
                  )}
                </div>

                {/* Repeated denied users */}
                <div>
                  <h4 className="text-sm font-bold text-hiplink-dark dark:text-dark-text mb-3">Repeated Permission Denied</h4>
                  {data.risky_activity.repeated_permission_denied_users.length > 0 ? (
                    <div className="space-y-2">
                      {data.risky_activity.repeated_permission_denied_users.map((item) => (
                        <div key={item.username} className="flex items-center justify-between py-2 px-3 rounded-lg bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20">
                          <span className="text-sm text-hiplink-dark dark:text-dark-text font-medium">{item.username}</span>
                          <span className="text-sm font-bold text-amber-600 dark:text-amber-400">{item.denied_count} denied</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No repeated permission denials detected.</p>
                  )}
                </div>
              </div>
            )}
          </Section>

          {/* Export */}
          <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl p-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h3 className="text-base font-bold text-hiplink-dark dark:text-dark-text">Export Audit Records</h3>
                <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted mt-0.5">
                  Download {hours === 24 ? 'last 24 hours' : hours === 168 ? 'last 7 days' : 'last 30 days'} as CSV or JSON.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleExport('csv')}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-hiplink-blue text-white text-sm font-medium hover:bg-hiplink-blue/90 transition-colors"
                >
                  {iconDownload}
                  Export CSV
                </button>
                <button
                  onClick={() => handleExport('json')}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border text-hiplink-dark dark:text-dark-text text-sm font-medium hover:bg-gray-50 dark:hover:bg-dark-border transition-colors"
                >
                  {iconDownload}
                  Export JSON
                </button>
              </div>
            </div>
          </div>

          {/* Security Policy & Controls */}
          <div className="bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-hiplink-border dark:border-dark-border">
              <h2 className="text-base font-bold text-hiplink-dark dark:text-dark-text">Security Policy &amp; Controls</h2>
              <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted mt-0.5">
                Read-only view of active security controls and policy settings.
              </p>
            </div>
            <div className="p-6 space-y-6">

              {/* Policy cards grid */}
              {loading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="rounded-xl border border-hiplink-border dark:border-dark-border p-5 space-y-3">
                      <SkeletonBar w="w-1/2" />
                      <SkeletonBar w="w-full" />
                      <SkeletonBar w="w-3/4" />
                    </div>
                  ))}
                </div>
              ) : data?.policy_controls ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

                  {/* Password Policy */}
                  <div className="rounded-xl border border-hiplink-border dark:border-dark-border p-5 bg-gray-50 dark:bg-dark-elevated">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-8 h-8 rounded-lg bg-sky-100 dark:bg-sky-900/30 flex items-center justify-center text-sky-600 dark:text-sky-400">
                        {iconKey}
                      </div>
                      <h3 className="text-sm font-bold text-hiplink-dark dark:text-dark-text">Password Policy</h3>
                    </div>
                    <div className="space-y-1.5 text-sm">
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Minimum Length</span>
                        <span className="font-medium text-hiplink-dark dark:text-dark-text">{data.policy_controls.password_policy.minimum_length}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Complexity Required</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">
                          {data.policy_controls.password_policy.complexity_enabled ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Weak Password Blocking</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">
                          {data.policy_controls.password_policy.weak_password_blocking_enabled ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Password Reset Support</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">
                          {data.policy_controls.password_policy.password_reset_support ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Session Management */}
                  <div className="rounded-xl border border-hiplink-border dark:border-dark-border p-5 bg-gray-50 dark:bg-dark-elevated">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-8 h-8 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 dark:text-purple-400">
                        {iconShield}
                      </div>
                      <h3 className="text-sm font-bold text-hiplink-dark dark:text-dark-text">Session Management</h3>
                    </div>
                    <div className="space-y-1.5 text-sm">
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Auth Mode</span>
                        <span className="font-medium text-hiplink-dark dark:text-dark-text">{data.policy_controls.session_management.auth_mode}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Token Version Invalidation</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">
                          {data.policy_controls.session_management.token_version_invalidation ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Active Users Count</span>
                        <span className="font-medium text-hiplink-secondary dark:text-dark-text-muted">{data.policy_controls.session_management.active_users_count}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Force Logout All</span>
                        <span className="font-medium text-amber-600 dark:text-amber-400">{data.policy_controls.session_management.force_logout_all_status}</span>
                      </div>
                    </div>
                  </div>

                  {/* Login Protection */}
                  <div className="rounded-xl border border-hiplink-border dark:border-dark-border p-5 bg-gray-50 dark:bg-dark-elevated">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-8 h-8 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center text-red-600 dark:text-red-400">
                        {iconLock}
                      </div>
                      <h3 className="text-sm font-bold text-hiplink-dark dark:text-dark-text">Login Protection</h3>
                    </div>
                    <div className="space-y-1.5 text-sm">
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Failed Login Audit</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">{data.policy_controls.login_protection.failed_login_audit_tracking}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Repeated IP Detection</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">{data.policy_controls.login_protection.repeated_ip_detection}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Account Lockout</span>
                        <span className="font-medium text-hiplink-secondary dark:text-dark-text-muted">{data.policy_controls.login_protection.account_lockout}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Failed Logins (24h)</span>
                        <span className="font-medium text-hiplink-dark dark:text-dark-text">{data.policy_controls.login_protection.failed_login_count_window}</span>
                      </div>
                    </div>
                  </div>

                  {/* Audit & Evidence */}
                  <div className="rounded-xl border border-hiplink-border dark:border-dark-border p-5 bg-gray-50 dark:bg-dark-elevated">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-8 h-8 rounded-lg bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center text-emerald-600 dark:text-emerald-400">
                        {iconDoc}
                      </div>
                      <h3 className="text-sm font-bold text-hiplink-dark dark:text-dark-text">Audit &amp; Evidence</h3>
                    </div>
                    <div className="space-y-1.5 text-sm">
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">CSV Export</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">{data.policy_controls.audit_evidence.csv_export}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">JSON Export</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">{data.policy_controls.audit_evidence.json_export}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Export Limit</span>
                        <span className="font-medium text-hiplink-dark dark:text-dark-text">{data.policy_controls.audit_evidence.export_limit.toLocaleString()} records</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Audit Logs Link</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">
                          {data.policy_controls.audit_evidence.audit_logs_link_available ? 'Available' : 'Unavailable'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* RAG Safety */}
                  <div className="rounded-xl border border-hiplink-border dark:border-dark-border p-5 bg-gray-50 dark:bg-dark-elevated">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-8 h-8 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 dark:text-purple-400">
                        {iconAlert}
                      </div>
                      <h3 className="text-sm font-bold text-hiplink-dark dark:text-dark-text">RAG Safety</h3>
                    </div>
                    <div className="space-y-1.5 text-sm">
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">CRAG Fallback Checks</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">{data.policy_controls.rag_safety.crag_fallback_checks}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">High-Risk Categories</span>
                        <span className="font-medium text-hiplink-dark dark:text-dark-text">
                          {data.policy_controls.rag_safety.high_risk_categories.join(', ')}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">RAG Evaluation</span>
                        <span className="font-medium text-hiplink-dark dark:text-dark-text">{data.policy_controls.rag_safety.rag_evaluation_status}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-hiplink-secondary dark:text-dark-text-muted">Unsupported Question</span>
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">{data.policy_controls.rag_safety.unsupported_question_fallback}</span>
                      </div>
                    </div>
                  </div>

                </div>
              ) : (
                <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">Policy controls data is not available.</p>
              )}

              {/* Quick links */}
              <div className="flex items-center justify-between flex-wrap gap-3 pt-2 border-t border-hiplink-border dark:border-dark-border">
                <div className="flex items-center gap-2 flex-wrap">
                  <a
                    href="/admin/users"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border text-hiplink-dark dark:text-dark-text text-sm font-medium hover:bg-gray-50 dark:hover:bg-dark-border transition-colors"
                  >
                    {iconUsers}
                    View Users
                  </a>
                  <a
                    href="/admin/audit-logs"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border text-hiplink-dark dark:text-dark-text text-sm font-medium hover:bg-gray-50 dark:hover:bg-dark-border transition-colors"
                  >
                    {iconDoc}
                    View Audit Logs
                  </a>
                  <button
                    disabled
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-100 dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border text-hiplink-secondary dark:text-dark-text-muted text-sm font-medium opacity-50 cursor-not-allowed"
                    title="Coming soon"
                  >
                    {iconUsers}
                    Force Logout All Users — Coming soon
                  </button>
                </div>
              </div>

            </div>
          </div>

        </div>
      </main>
    </ProtectedRoute>
  )
}

export default SecurityPage
