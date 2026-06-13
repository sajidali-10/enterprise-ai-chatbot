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

  if (loading) {
    return <div className="text-center text-gray-500 py-8">Loading observability data...</div>
  }

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
        <strong>Error:</strong> {error}
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Observability Dashboard</h2>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.total_questions}</div>
            <div className="text-sm text-gray-500">Total Questions</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.answered_count}</div>
            <div className="text-sm text-gray-500">Answered</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.blocked_count}</div>
            <div className="text-sm text-gray-500">Blocked</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.blocked_rate.toFixed(1)}%</div>
            <div className="text-sm text-gray-500">Blocked Rate</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.citation_rate.toFixed(1)}%</div>
            <div className="text-sm text-gray-500">Citation Rate</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{formatLatency(summary.average_latency_ms)}</div>
            <div className="text-sm text-gray-500">Avg Latency</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.thumbs_up_count}</div>
            <div className="text-sm text-gray-500">👍 Helpful</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.thumbs_down_count}</div>
            <div className="text-sm text-gray-500">👎 Not Helpful</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.average_top_score.toFixed(3)}</div>
            <div className="text-sm text-gray-500">Avg Top Score</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.knowledge_base_count}</div>
            <div className="text-sm text-gray-500">Knowledge Base</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.general_chat_count}</div>
            <div className="text-sm text-gray-500">General Chat</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">{summary.debug_count}</div>
            <div className="text-sm text-gray-500">Debug Mode</div>
          </div>
        </div>
      )}

      {/* Most Used Sources */}
      {summary && summary.most_used_sources.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-gray-900 mb-3">Most Used Source Documents</h3>
          <div className="flex flex-wrap gap-2">
            {summary.most_used_sources.map((src, i) => (
              <span key={i} className="bg-gray-100 text-gray-700 px-3 py-1 rounded-full text-sm">
                {src.source} ({src.count})
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex space-x-2 mb-4">
        <button
          onClick={() => setActiveTab('recent')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'recent'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Recent Questions
        </button>
        <button
          onClick={() => setActiveTab('blocked')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'blocked'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Blocked
        </button>
        <button
          onClick={() => setActiveTab('low-confidence')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'low-confidence'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Low Confidence
        </button>
      </div>

      {/* Recent Questions Table */}
      {activeTab === 'recent' && (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Time</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">User</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Mode</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Question</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Sources</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Citations</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Latency</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Feedback</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {recentObs.map((obs) => (
                <tr key={obs.id} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-xs text-gray-500 whitespace-nowrap">{formatDate(obs.created_at)}</td>
                  <td className="px-3 py-2 text-xs text-gray-700">{obs.username || 'Anonymous'}</td>
                  <td className="px-3 py-2 text-xs">
                    <span className={`px-2 py-0.5 rounded-full text-xs ${
                      obs.mode === 'general_chat' ? 'bg-blue-100 text-blue-700' :
                      obs.mode === 'knowledge_base' ? 'bg-green-100 text-green-700' :
                      'bg-purple-100 text-purple-700'
                    }`}>
                      {obs.mode}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-900 max-w-xs truncate">{obs.question}</td>
                  <td className="px-3 py-2 text-xs text-gray-600">
                    {obs.source_files ? obs.source_files.length : 0}
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-600">{obs.citation_count || 0}</td>
                  <td className="px-3 py-2 text-xs text-gray-600">{formatLatency(obs.latency_ms)}</td>
                  <td className="px-3 py-2 text-xs">
                    {obs.feedback_rating === 'helpful' && '👍'}
                    {obs.feedback_rating === 'not_helpful' && '👎'}
                    {!obs.feedback_rating && '—'}
                  </td>
                </tr>
              ))}
              {recentObs.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-3 py-4 text-center text-gray-500 text-sm">No observations yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Blocked Questions Table */}
      {activeTab === 'blocked' && (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Time</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Question</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Block Reason</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Top Score</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Mode</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {blockedObs.map((obs) => (
                <tr key={obs.id} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-xs text-gray-500 whitespace-nowrap">{formatDate(obs.created_at)}</td>
                  <td className="px-3 py-2 text-xs text-gray-900 max-w-xs truncate">{obs.question}</td>
                  <td className="px-3 py-2 text-xs text-red-600">{obs.block_reason || 'Insufficient information'}</td>
                  <td className="px-3 py-2 text-xs text-gray-600">{obs.top_score?.toFixed(3) || 'N/A'}</td>
                  <td className="px-3 py-2 text-xs">
                    <span className={`px-2 py-0.5 rounded-full text-xs ${
                      obs.mode === 'general_chat' ? 'bg-blue-100 text-blue-700' :
                      obs.mode === 'knowledge_base' ? 'bg-green-100 text-green-700' :
                      'bg-purple-100 text-purple-700'
                    }`}>
                      {obs.mode}
                    </span>
                  </td>
                </tr>
              ))}
              {blockedObs.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-3 py-4 text-center text-gray-500 text-sm">No blocked questions.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Low Confidence Table */}
      {activeTab === 'low-confidence' && (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Time</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Question</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Top Score</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Citations</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Feedback</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Block Reason</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {lowConfObs.map((obs) => (
                <tr key={obs.id} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-xs text-gray-500 whitespace-nowrap">{formatDate(obs.created_at)}</td>
                  <td className="px-3 py-2 text-xs text-gray-900 max-w-xs truncate">{obs.question}</td>
                  <td className="px-3 py-2 text-xs">
                    <span className={obs.top_score && obs.top_score < 0.5 ? 'text-red-600 font-medium' : 'text-gray-600'}>
                      {obs.top_score?.toFixed(3) || 'N/A'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-600">{obs.citation_count || 0}</td>
                  <td className="px-3 py-2 text-xs">
                    {obs.feedback_rating === 'helpful' && '👍'}
                    {obs.feedback_rating === 'not_helpful' && '👎'}
                    {!obs.feedback_rating && '—'}
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-600">{obs.block_reason || '—'}</td>
                </tr>
              ))}
              {lowConfObs.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-3 py-4 text-center text-gray-500 text-sm">No low confidence observations.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}