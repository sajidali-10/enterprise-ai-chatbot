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

interface MetricCardProps {
  label: string
  value: string | number
  highlight?: 'success' | 'warning' | 'error' | 'blue'
}

function MetricCard({ label, value, highlight }: MetricCardProps) {
  const colorClasses = {
    success: 'text-hiplink-success',
    warning: 'text-hiplink-warning',
    error: 'text-hiplink-error',
    blue: 'text-hiplink-blue',
  }

  return (
    <div className="card p-4">
      <div className={`text-2xl font-bold ${highlight ? colorClasses[highlight] : 'text-hiplink-dark'}`}>
        {value}
      </div>
      <div className="text-sm text-hiplink-secondary mt-1">{label}</div>
    </div>
  )
}

export default function ObservabilityPage() {
  const [summary, setSummary] = useState<ObservabilitySummary | null>(null)
  const [recentObs, setRecentObs] = useState<Observation[]>([])
  const [blockedObs, setBlockedObs] = useState<BlockedObservation[]>([])
  const [lowConfObs, setLowConfObs] = useState<LowConfidenceObservation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'recent' | 'blocked' | 'low-confidence'>('recent')
  const authFetch = useAuthFetch()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [summaryRes, recentRes, blockedRes, lowConfRes] = await Promise.all([
        authFetch('/api/admin/observability/summary'),
        authFetch('/api/admin/observability/recent?limit=50'),
        authFetch('/api/admin/observability/blocked?limit=50'),
        authFetch('/api/admin/observability/low-confidence?limit=50'),
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
        return 'bg-blue-100 text-hiplink-blue'
      case 'knowledge_base':
        return 'bg-green-100 text-hiplink-success'
      case 'debug':
        return 'bg-purple-100 text-purple-700'
      default:
        return 'bg-gray-100 text-hiplink-secondary'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-hiplink-secondary">Loading observability data...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-6 border-hiplink-error">
        <strong className="text-hiplink-error">Error:</strong> {error}
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-xl font-semibold text-hiplink-dark mb-6">Observability Dashboard</h2>

      {/* Summary Cards */}
      {summary && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            <MetricCard label="Total Questions" value={summary.total_questions} />
            <MetricCard label="Answered" value={summary.answered_count} highlight="success" />
            <MetricCard label="Blocked" value={summary.blocked_count} highlight="error" />
            <MetricCard label="Blocked Rate" value={`${summary.blocked_rate.toFixed(1)}%`} highlight={summary.blocked_rate > 10 ? 'warning' : undefined} />
            <MetricCard label="Citation Rate" value={`${summary.citation_rate.toFixed(1)}%`} highlight="blue" />
            <MetricCard label="Avg Latency" value={formatLatency(summary.average_latency_ms)} />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard label="👍 Helpful" value={summary.thumbs_up_count} highlight="success" />
            <MetricCard label="👎 Not Helpful" value={summary.thumbs_down_count} highlight="warning" />
            <MetricCard label="Avg Top Score" value={summary.average_top_score.toFixed(3)} />
            <MetricCard label="Knowledge Base" value={summary.knowledge_base_count} />
            <MetricCard label="General Chat" value={summary.general_chat_count} />
            <MetricCard label="Debug Mode" value={summary.debug_count} />
          </div>
        </>
      )}

      {/* Most Used Sources */}
      {summary && summary.most_used_sources.length > 0 && (
        <div className="card p-4 mb-6">
          <h3 className="font-semibold text-hiplink-dark mb-3">Most Used Source Documents</h3>
          <div className="flex flex-wrap gap-2">
            {summary.most_used_sources.map((src, i) => (
              <span key={i} className="bg-blue-50 text-hiplink-blue px-3 py-1 rounded-full text-sm">
                {src.source} ({src.count})
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex space-x-2 mb-4">
        {(['recent', 'blocked', 'low-confidence'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === tab
                ? 'bg-hiplink-blue text-white'
                : 'bg-white text-hiplink-dark border border-hiplink-border hover:bg-gray-50'
            }`}
          >
            {tab === 'recent' ? 'Recent Questions' : tab === 'blocked' ? 'Blocked' : 'Low Confidence'}
          </button>
        ))}
      </div>

      {/* Tables */}
      <div className="card overflow-hidden">
        {/* Recent Questions Table */}
        {activeTab === 'recent' && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-hiplink-border">
              <thead className="bg-hiplink-background">
                <tr>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Time</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">User</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Mode</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Question</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Sources</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Citations</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Latency</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Feedback</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hiplink-border">
                {recentObs.map((obs) => (
                  <tr key={obs.id} className="hover:bg-hiplink-background transition-colors">
                    <td className="px-3 py-3 text-xs text-hiplink-secondary whitespace-nowrap">{formatDate(obs.created_at)}</td>
                    <td className="px-3 py-3 text-xs text-hiplink-dark">{obs.username || 'Anonymous'}</td>
                    <td className="px-3 py-3 text-xs">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getModeBadgeClass(obs.mode)}`}>
                        {obs.mode}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-xs text-hiplink-dark max-w-xs truncate">{obs.question}</td>
                    <td className="px-3 py-3 text-xs text-hiplink-secondary">{obs.source_files?.length || 0}</td>
                    <td className="px-3 py-3 text-xs text-hiplink-secondary">{obs.citation_count || 0}</td>
                    <td className="px-3 py-3 text-xs text-hiplink-secondary">{formatLatency(obs.latency_ms)}</td>
                    <td className="px-3 py-3 text-xs">
                      {obs.feedback_rating === 'helpful' && '👍'}
                      {obs.feedback_rating === 'not_helpful' && '👎'}
                      {!obs.feedback_rating && '—'}
                    </td>
                  </tr>
                ))}
                {recentObs.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-3 py-8 text-center text-hiplink-secondary text-sm">No observations yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Blocked Questions Table */}
        {activeTab === 'blocked' && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-hiplink-border">
              <thead className="bg-hiplink-background">
                <tr>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Time</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Question</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Block Reason</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Top Score</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Mode</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hiplink-border">
                {blockedObs.map((obs) => (
                  <tr key={obs.id} className="hover:bg-hiplink-background transition-colors">
                    <td className="px-3 py-3 text-xs text-hiplink-secondary whitespace-nowrap">{formatDate(obs.created_at)}</td>
                    <td className="px-3 py-3 text-xs text-hiplink-dark max-w-xs truncate">{obs.question}</td>
                    <td className="px-3 py-3 text-xs text-hiplink-error">{obs.block_reason || 'Insufficient information'}</td>
                    <td className="px-3 py-3 text-xs text-hiplink-secondary">{obs.top_score?.toFixed(3) || 'N/A'}</td>
                    <td className="px-3 py-3 text-xs">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getModeBadgeClass(obs.mode)}`}>
                        {obs.mode}
                      </span>
                    </td>
                  </tr>
                ))}
                {blockedObs.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-hiplink-secondary text-sm">No blocked questions.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Low Confidence Table */}
        {activeTab === 'low-confidence' && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-hiplink-border">
              <thead className="bg-hiplink-background">
                <tr>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Time</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Question</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Top Score</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Citations</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Feedback</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Block Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hiplink-border">
                {lowConfObs.map((obs) => (
                  <tr key={obs.id} className="hover:bg-hiplink-background transition-colors">
                    <td className="px-3 py-3 text-xs text-hiplink-secondary whitespace-nowrap">{formatDate(obs.created_at)}</td>
                    <td className="px-3 py-3 text-xs text-hiplink-dark max-w-xs truncate">{obs.question}</td>
                    <td className="px-3 py-3 text-xs">
                      <span className={obs.top_score && obs.top_score < 0.5 ? 'text-hiplink-error font-medium' : 'text-hiplink-secondary'}>
                        {obs.top_score?.toFixed(3) || 'N/A'}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-xs text-hiplink-secondary">{obs.citation_count || 0}</td>
                    <td className="px-3 py-3 text-xs">
                      {obs.feedback_rating === 'helpful' && '👍'}
                      {obs.feedback_rating === 'not_helpful' && '👎'}
                      {!obs.feedback_rating && '—'}
                    </td>
                    <td className="px-3 py-3 text-xs text-hiplink-secondary">{obs.block_reason || '—'}</td>
                  </tr>
                ))}
                {lowConfObs.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-3 py-8 text-center text-hiplink-secondary text-sm">No low confidence observations.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}