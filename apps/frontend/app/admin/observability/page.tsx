'use client'

import { useState, useEffect } from 'react'
import { useAuthFetch } from '@/hooks/useApi'

interface ObservabilitySummary {
  total_questions: number
  general_chat_count: number
  knowledge_base_count: number
  debug_count: number
  answered_count: number
  blocked_count: number
  blocked_rate: number
  citation_rate: number
  average_latency_ms: number
  average_top_score: number
  thumbs_up_count: number
  thumbs_down_count: number
  most_used_sources: Array<{ source: string; count: number }>
}

interface Observation {
  id: number
  created_at: string
  username: string | null
  mode: string
  question: string
  answer_preview: string | null
  answer_length: number | null
  source_files: string[] | null
  citation_count: number | null
  top_score: number | null
  blocked: boolean
  block_reason: string | null
  latency_ms: number | null
  feedback_rating: string | null
}

interface BlockedObservation {
  id: number
  created_at: string
  question: string
  block_reason: string | null
  top_score: number | null
  mode: string
}

interface LowConfidenceObservation {
  id: number
  created_at: string
  question: string
  top_score: number | null
  citation_count: number | null
  feedback_rating: string | null
  block_reason: string | null
}

interface LangSmithSummary {
  available: boolean
  tracing_enabled: boolean
  project: string
  endpoint_host: string
  has_tracing_key: boolean
  sample_rate: number
  log_full_prompt: boolean
  log_document_text: boolean
  log_user_input: boolean
  log_retrieved_context: boolean
  privacy_mode: string
  warnings: string[]
}

interface MetricCardProps {
  label: string
  value: string | number
  highlight?: 'success' | 'warning' | 'error' | 'blue' | 'neutral'
  icon?: React.ReactNode
}

function MetricCard({ label, value, highlight, icon }: MetricCardProps) {
  const colorClasses = {
    success: 'text-hiplink-success dark:text-green-400',
    warning: 'text-hiplink-warning dark:text-amber-400',
    error: 'text-hiplink-error dark:text-red-400',
    blue: 'text-hiplink-blue dark:text-sky-400',
    neutral: 'text-hiplink-dark dark:text-dark-text',
  }

  return (
    <div className="card dark:bg-dark-card p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div>
          <div className={`text-2xl font-bold ${highlight ? colorClasses[highlight] : 'text-hiplink-dark dark:text-dark-text'}`}>
            {value}
          </div>
          <div className="text-sm text-hiplink-secondary dark:text-dark-text-dim mt-1">{label}</div>
        </div>
        {icon && (
          <div className="text-hiplink-secondary dark:text-dark-text-dim opacity-50">
            {icon}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ObservabilityPage() {
  const [summary, setSummary] = useState<ObservabilitySummary | null>(null)
  const [recentObs, setRecentObs] = useState<Observation[]>([])
  const [blockedObs, setBlockedObs] = useState<BlockedObservation[]>([])
  const [lowConfObs, setLowConfObs] = useState<LowConfidenceObservation[]>([])
  const [langSmithSummary, setLangSmithSummary] = useState<LangSmithSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'recent' | 'blocked' | 'low-confidence' | 'langsmith'>('recent')
  const authFetch = useAuthFetch()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [summaryRes, recentRes, blockedRes, lowConfRes, langSmithRes] = await Promise.all([
        authFetch('/api/admin/observability/summary'),
        authFetch('/api/admin/observability/recent?limit=50'),
        authFetch('/api/admin/observability/blocked?limit=50'),
        authFetch('/api/admin/observability/low-confidence?limit=50'),
        authFetch('/api/admin/observability/langsmith-summary'),
      ])

      if (!summaryRes.ok || !recentRes.ok || !blockedRes.ok || !lowConfRes.ok) {
        throw new Error('Failed to fetch observability data')
      }

      const [summaryData, recentData, blockedData, lowConfData] = await Promise.all([
        summaryRes.json(),
        recentRes.json(),
        blockedRes.json(),
        lowConfRes.json(),
      ])

      setSummary(summaryData)
      setRecentObs(recentData)
      setBlockedObs(blockedData)
      setLowConfObs(lowConfData)

      if (langSmithRes.ok) {
        setLangSmithSummary(await langSmithRes.json())
      }
      setBlockedObs(blockedData)
      setLowConfObs(lowConfData)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString()
  }

  const formatLatency = (ms: number | null) => {
    if (ms === null) return 'N/A'
    return `${ms.toFixed(0)}ms`
  }

  const getModeBadgeClass = (mode: string) => {
    switch (mode) {
      case 'general_chat':
        return 'bg-blue-100 dark:bg-blue-900/30 text-hiplink-blue dark:text-blue-400'
      case 'knowledge_base':
        return 'bg-green-100 dark:bg-green-900/30 text-hiplink-success dark:text-green-400'
      case 'debug':
        return 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400'
      default:
        return 'bg-gray-100 dark:bg-gray-800 text-hiplink-secondary dark:text-dark-text-dim'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-hiplink-secondary dark:text-dark-text-dim">Loading observability data...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card dark:bg-dark-card p-6 border-hiplink-error dark:border-red-500">
        <strong className="text-hiplink-error dark:text-red-400">Error:</strong> {error}
      </div>
    )
  }

  return (
    <div>
      {/* Page Header */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-hiplink-dark dark:text-dark-text">Observability Dashboard</h2>
        <p className="text-hiplink-secondary dark:text-dark-text-dim mt-1">Monitor usage, latency, feedback, blocked answers, and source activity.</p>
      </div>

      {/* Summary Cards */}
      {summary && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            <MetricCard 
              label="Total Questions" 
              value={summary.total_questions} 
              icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>}
            />
            <MetricCard 
              label="Answered" 
              value={summary.answered_count} 
              highlight="success"
              icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>}
            />
            <MetricCard 
              label="Blocked" 
              value={summary.blocked_count} 
              highlight="error"
              icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" /></svg>}
            />
            <MetricCard 
              label="Blocked Rate" 
              value={`${summary.blocked_rate.toFixed(1)}%`} 
              highlight={summary.blocked_rate > 10 ? 'warning' : undefined}
              icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>}
            />
            <MetricCard 
              label="Citation Rate" 
              value={`${summary.citation_rate.toFixed(1)}%`} 
              highlight="blue"
              icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>}
            />
            <MetricCard 
              label="Avg Latency" 
              value={formatLatency(summary.average_latency_ms)}
              icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
            />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard 
              label="👍 Helpful" 
              value={summary.thumbs_up_count} 
              highlight="success"
            />
            <MetricCard 
              label="👎 Not Helpful" 
              value={summary.thumbs_down_count} 
              highlight="warning"
            />
            <MetricCard 
              label="Avg Top Score" 
              value={summary.average_top_score.toFixed(3)} 
            />
            <MetricCard 
              label="Knowledge Base" 
              value={summary.knowledge_base_count} 
              highlight="blue"
            />
            <MetricCard 
              label="General Chat" 
              value={summary.general_chat_count} 
            />
            <MetricCard 
              label="Debug Mode" 
              value={summary.debug_count} 
            />
          </div>
        </>
      )}

      {/* Most Used Sources */}
      {summary && summary.most_used_sources.length > 0 && (
        <div className="card dark:bg-dark-card p-4 mb-6">
          <h3 className="font-semibold text-hiplink-dark dark:text-dark-text mb-3 flex items-center gap-2">
            <svg className="w-5 h-5 text-hiplink-secondary dark:text-dark-text-dim" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Most Used Source Documents
          </h3>
          <div className="flex flex-wrap gap-2">
            {summary.most_used_sources.map((src, i) => (
              <span key={i} className="bg-blue-50 dark:bg-sky-900/30 text-hiplink-blue dark:text-sky-400 px-3 py-1.5 rounded-full text-sm font-medium">
                {src.source} <span className="text-blue-300 dark:text-sky-600">({src.count})</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex space-x-2 mb-4">
        {([
          { key: 'recent' as const, label: 'Recent Questions', icon: '📋' },
          { key: 'blocked' as const, label: 'Blocked', icon: '🚫' },
          { key: 'low-confidence' as const, label: 'Low Confidence', icon: '⚠️' },
          { key: 'langsmith' as const, label: 'LangSmith Tracing', icon: '🔍' },
        ]).map(({ key, label, icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`px-4 py-2.5 rounded-lg font-medium transition-colors flex items-center gap-2 ${
              activeTab === key
                ? 'bg-hiplink-blue dark:bg-sky-600 text-white'
                : 'bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text border border-hiplink-border dark:border-dark-border hover:bg-gray-50 dark:hover:bg-dark-elevated'
            }`}
          >
            <span>{icon}</span>
            {label}
          </button>
        ))}
      </div>

      {/* Tables */}
      <div className="card dark:bg-dark-card overflow-hidden">
        {/* Recent Questions Table */}
        {activeTab === 'recent' && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-hiplink-border dark:divide-dark-border">
              <thead className="bg-hiplink-background dark:bg-dark-elevated">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Time</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">User</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Mode</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Question</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Sources</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Citations</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Latency</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Feedback</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                {recentObs.map((obs) => (
                  <tr key={obs.id} className="hover:bg-hiplink-background dark:hover:bg-dark-elevated transition-colors">
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim whitespace-nowrap">{formatDate(obs.created_at)}</td>
                    <td className="px-4 py-3 text-xs text-hiplink-dark dark:text-dark-text">{obs.username || 'Anonymous'}</td>
                    <td className="px-4 py-3 text-xs">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getModeBadgeClass(obs.mode)}`}>
                        {obs.mode}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-hiplink-dark dark:text-dark-text max-w-xs truncate">{obs.question}</td>
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{obs.source_files?.length || 0}</td>
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{obs.citation_count || 0}</td>
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{formatLatency(obs.latency_ms)}</td>
                    <td className="px-4 py-3 text-xs">
                      {obs.feedback_rating === 'helpful' && '👍'}
                      {obs.feedback_rating === 'not_helpful' && '👎'}
                      {!obs.feedback_rating && '—'}
                    </td>
                  </tr>
                ))}
                {recentObs.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-hiplink-secondary dark:text-dark-text-dim text-sm">No observations yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Blocked Questions Table */}
        {activeTab === 'blocked' && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-hiplink-border dark:divide-dark-border">
              <thead className="bg-hiplink-background dark:bg-dark-elevated">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Time</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Question</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Block Reason</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Top Score</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Mode</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                {blockedObs.map((obs) => (
                  <tr key={obs.id} className="hover:bg-hiplink-background dark:hover:bg-dark-elevated transition-colors">
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim whitespace-nowrap">{formatDate(obs.created_at)}</td>
                    <td className="px-4 py-3 text-xs text-hiplink-dark dark:text-dark-text max-w-xs truncate">{obs.question}</td>
                    <td className="px-4 py-3 text-xs text-hiplink-error dark:text-red-400">{obs.block_reason || 'Insufficient information'}</td>
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{obs.top_score?.toFixed(3) || 'N/A'}</td>
                    <td className="px-4 py-3 text-xs">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getModeBadgeClass(obs.mode)}`}>
                        {obs.mode}
                      </span>
                    </td>
                  </tr>
                ))}
                {blockedObs.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-8 text-center text-hiplink-secondary dark:text-dark-text-dim text-sm">No blocked questions.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Low Confidence Table */}
        {activeTab === 'low-confidence' && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-hiplink-border dark:divide-dark-border">
              <thead className="bg-hiplink-background dark:bg-dark-elevated">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Time</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Question</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Top Score</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Citations</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Feedback</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Block Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                {lowConfObs.map((obs) => (
                  <tr key={obs.id} className="hover:bg-hiplink-background dark:hover:bg-dark-elevated transition-colors">
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim whitespace-nowrap">{formatDate(obs.created_at)}</td>
                    <td className="px-4 py-3 text-xs text-hiplink-dark dark:text-dark-text max-w-xs truncate">{obs.question}</td>
                    <td className="px-4 py-3 text-xs">
                      <span className={obs.top_score && obs.top_score < 0.5 ? 'text-hiplink-error dark:text-red-400 font-medium' : 'text-hiplink-secondary dark:text-dark-text-dim'}>
                        {obs.top_score?.toFixed(3) || 'N/A'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{obs.citation_count || 0}</td>
                    <td className="px-4 py-3 text-xs">
                      {obs.feedback_rating === 'helpful' && '👍'}
                      {obs.feedback_rating === 'not_helpful' && '👎'}
                      {!obs.feedback_rating && '—'}
                    </td>
                    <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{obs.block_reason || '—'}</td>
                  </tr>
                ))}
                {lowConfObs.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-hiplink-secondary dark:text-dark-text-dim text-sm">No low confidence observations.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* LangSmith Tracing Tab */}
        {activeTab === 'langsmith' && (
          <div className="p-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <MetricCard
                label="Tracing Status"
                value={!langSmithSummary ? 'Loading…' : langSmithSummary.tracing_enabled ? 'Enabled' : 'Disabled'}
                highlight={!langSmithSummary ? 'neutral' : langSmithSummary.tracing_enabled ? 'success' : 'warning'}
              />
              <MetricCard
                label="LangSmith Availability"
                value={!langSmithSummary ? '—' : langSmithSummary.available ? 'Available' : 'Not Installed'}
                highlight={!langSmithSummary ? 'neutral' : langSmithSummary.available ? 'success' : 'warning'}
              />
              <MetricCard
                label="Project"
                value={langSmithSummary?.project || '—'}
              />
              <MetricCard
                label="Endpoint"
                value={langSmithSummary?.endpoint_host || '—'}
              />
              <MetricCard
                label="API Key"
                value={langSmithSummary?.has_tracing_key ? 'Configured' : 'Not Configured'}
                highlight={langSmithSummary?.has_tracing_key ? 'success' : 'warning'}
              />
              <MetricCard
                label="Sample Rate"
                value={langSmithSummary?.sample_rate?.toString() || '—'}
              />
              <MetricCard
                label="Privacy Mode"
                value={langSmithSummary?.privacy_mode || '—'}
                highlight={langSmithSummary?.privacy_mode === 'Safe' ? 'success' : langSmithSummary?.privacy_mode === 'Restricted' ? 'warning' : 'neutral'}
              />
            </div>

            {/* Privacy Controls */}
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text mb-3">
                Privacy Controls
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className={`p-3 rounded-lg border ${langSmithSummary?.log_full_prompt ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20' : 'border-hiplink-border dark:border-dark-border bg-hiplink-background dark:bg-dark-elevated'}`}>
                  <div className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-1">Full Prompt Logging</div>
                  <div className={`text-sm font-semibold ${langSmithSummary?.log_full_prompt ? 'text-hiplink-success dark:text-green-400' : 'text-hiplink-secondary dark:text-dark-text-dim'}`}>
                    {langSmithSummary?.log_full_prompt ? 'Enabled' : 'Disabled'}
                  </div>
                </div>
                <div className={`p-3 rounded-lg border ${langSmithSummary?.log_document_text ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20' : 'border-hiplink-border dark:border-dark-border bg-hiplink-background dark:bg-dark-elevated'}`}>
                  <div className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-1">Document Text Logging</div>
                  <div className={`text-sm font-semibold ${langSmithSummary?.log_document_text ? 'text-hiplink-success dark:text-green-400' : 'text-hiplink-secondary dark:text-dark-text-dim'}`}>
                    {langSmithSummary?.log_document_text ? 'Enabled' : 'Disabled'}
                  </div>
                </div>
                <div className={`p-3 rounded-lg border ${langSmithSummary?.log_user_input ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20' : 'border-hiplink-border dark:border-dark-border bg-hiplink-background dark:bg-dark-elevated'}`}>
                  <div className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-1">User Input Logging</div>
                  <div className={`text-sm font-semibold ${langSmithSummary?.log_user_input ? 'text-hiplink-success dark:text-green-400' : 'text-hiplink-secondary dark:text-dark-text-dim'}`}>
                    {langSmithSummary?.log_user_input ? 'Enabled' : 'Disabled'}
                  </div>
                </div>
                <div className={`p-3 rounded-lg border ${langSmithSummary?.log_retrieved_context ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20' : 'border-hiplink-border dark:border-dark-border bg-hiplink-background dark:bg-dark-elevated'}`}>
                  <div className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-1">Retrieved Context Logging</div>
                  <div className={`text-sm font-semibold ${langSmithSummary?.log_retrieved_context ? 'text-hiplink-success dark:text-green-400' : 'text-hiplink-secondary dark:text-dark-text-dim'}`}>
                    {langSmithSummary?.log_retrieved_context ? 'Enabled' : 'Disabled'}
                  </div>
                </div>
              </div>
            </div>

            {/* Warnings */}
            {langSmithSummary?.warnings && langSmithSummary.warnings.length > 0 && (
              <div className="mb-6 card dark:bg-dark-card p-4 border border-yellow-200 dark:border-yellow-800">
                <h3 className="text-sm font-semibold text-yellow-700 dark:text-yellow-400 mb-2">Notes</h3>
                <ul className="space-y-1">
                  {langSmithSummary.warnings.map((w, i) => (
                    <li key={i} className="text-xs text-yellow-700 dark:text-yellow-300 flex items-start gap-2">
                      <span className="mt-0.5">•</span>
                      <span>{w}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Disabled state */}
            {langSmithSummary && !langSmithSummary.tracing_enabled && (
              <div className="card dark:bg-dark-card p-6 text-center border border-blue-200 dark:border-blue-800">
                <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
                  <svg className="w-7 h-7 text-blue-500 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-hiplink-secondary dark:text-dark-text mb-1">
                  LangSmith tracing is currently disabled.
                </p>
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim max-w-md mx-auto">
                  Enable LANGSMITH_TRACING=true and configure LANGSMITH_API_KEY when ready to send traces.
                </p>
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mt-3">
                  LangSmith tracing captures execution metadata for debugging and performance analysis. Sensitive content logging is controlled by privacy flags and is disabled by default where applicable.
                </p>
              </div>
            )}

            {/* Enabled state */}
            {langSmithSummary && langSmithSummary.tracing_enabled && langSmithSummary.has_tracing_key && (
              <div className="card dark:bg-dark-card p-6 text-center border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20">
                <div className="w-14 h-14 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
                  <svg className="w-7 h-7 text-green-500 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-hiplink-success dark:text-green-400 mb-1">
                  LangSmith tracing is active.
                </p>
                <p className="text-xs text-green-700 dark:text-green-300 max-w-md mx-auto">
                  Traces are being captured and sent to the configured LangSmith project. Review trace data at api.smith.langchain.com.
                </p>
                <p className="text-xs text-green-700 dark:text-green-300 mt-3">
                  LangSmith tracing captures execution metadata for debugging and performance analysis. Sensitive content logging is controlled by privacy flags and is disabled by default where applicable.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}