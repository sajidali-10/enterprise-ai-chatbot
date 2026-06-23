'use client'

import { useState, useEffect } from 'react'
import { useAuthFetch } from '@/hooks/useApi'

interface EvaluationRunSummary {
  run_id: number
  created_at: string
  total_tests: number
  passed_tests: number
  failed_tests: number
  pass_percentage: number
  average_latency_ms: number | null
  average_top_score: number | null
  status: string
  notes: string | null
}

interface EvaluationResultDetail {
  test_case_id: string
  question: string
  expected_source_file: string | null
  actual_source_files: string[] | null
  expected_keywords: string[] | null
  found_keywords: string[] | null
  missing_keywords: string[] | null
  forbidden_keywords: string[] | null
  forbidden_keywords_found: string[] | null
  should_answer: boolean
  expected_fallback: boolean
  actual_blocked: boolean
  answer_preview: string | null
  citation_count: number
  top_score: number | null
  latency_ms: number
  passed: boolean
  failure_reasons: string[] | null
}

interface EvaluationLatestResponse {
  latest_run: EvaluationRunSummary | null
  failed_cases: EvaluationResultDetail[]
  pass_percentage: number
  average_latency_ms: number | null
  average_top_score: number | null
  timestamp: string | null
}

interface EvaluationRunDetail {
  summary: EvaluationRunSummary
  results: EvaluationResultDetail[]
  failed_results: EvaluationResultDetail[]
}

interface MetricCardProps {
  label: string
  value: string | number
  highlight?: 'success' | 'error' | 'warning' | 'blue' | 'neutral'
}

interface RAGASScoreMetrics {
  faithfulness: number | null
  answer_relevancy: number | null
  context_precision: number | null
  context_recall: number | null
  answer_correctness: number | null
}

interface RAGASSummary {
  available: boolean
  enabled: boolean
  evaluator_provider: string
  evaluator_model: string
  report_dir_configured: boolean
  latest_report_found: boolean
  latest_report_name: string | null
  latest_timestamp: string | null
  metrics: RAGASScoreMetrics | null
  skipped_metrics: string[]
  threshold_faithfulness: number
  threshold_answer_relevancy: number
  threshold_context_precision: number
  warnings: string[]
}

function MetricCard({ label, value, highlight }: MetricCardProps) {
  const colorClasses = {
    success: 'text-hiplink-success dark:text-green-400',
    error: 'text-hiplink-error dark:text-red-400',
    warning: 'text-hiplink-warning dark:text-amber-400',
    blue: 'text-hiplink-blue dark:text-sky-400',
    neutral: 'text-hiplink-dark dark:text-dark-text',
  }

  return (
    <div className="card dark:bg-dark-card p-5 hover:shadow-md transition-shadow">
      <div className={`text-2xl font-bold ${highlight ? colorClasses[highlight] : 'text-hiplink-dark dark:text-dark-text'}`}>
        {value}
      </div>
      <div className="text-sm text-hiplink-secondary dark:text-dark-text-dim mt-1">{label}</div>
    </div>
  )
}

interface RAGASMetricCardProps {
  name: string
  purpose: string
  score: number | null
  threshold: number
  skipped: boolean
  skippedReason?: string
}

function RAGASMetricCard({ name, purpose, score, threshold, skipped, skippedReason }: RAGASMetricCardProps) {
  const getStatusDisplay = () => {
    if (skipped || skippedReason) {
      return {
        badge: 'Skipped',
        badgeClass: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400',
        valueDisplay: skippedReason || 'ground_truth required',
        valueClass: 'text-amber-600 dark:text-amber-400',
      }
    }
    if (score === null) {
      return {
        badge: 'Pending',
        badgeClass: 'bg-gray-100 dark:bg-dark-elevated text-hiplink-secondary dark:text-dark-text-muted',
        valueDisplay: 'No report',
        valueClass: 'text-hiplink-secondary dark:text-dark-text-dim',
      }
    }
    const pass = score >= threshold
    return {
      badge: pass ? 'Pass' : 'Fail',
      badgeClass: pass
        ? 'bg-green-100 dark:bg-green-900/30 text-hiplink-success dark:text-green-400'
        : 'bg-red-100 dark:bg-red-900/30 text-hiplink-error dark:text-red-400',
      valueDisplay: score.toFixed(4),
      valueClass: pass ? 'text-hiplink-success dark:text-green-400' : 'text-hiplink-error dark:text-red-400',
    }
  }

  const status = getStatusDisplay()

  return (
    <div className="card dark:bg-dark-card p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-semibold text-hiplink-dark dark:text-dark-text text-sm">{name}</h4>
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${status.badgeClass}`}>
          {status.badge}
        </span>
      </div>
      <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-3">{purpose}</p>
      <div className="flex items-baseline gap-1">
        <span className={`text-xl font-bold ${status.valueClass}`}>
          {status.valueDisplay}
        </span>
        {!skipped && !skippedReason && score !== null && (
          <span className="text-xs text-hiplink-secondary dark:text-dark-text-dim">
            / {threshold} threshold
          </span>
        )}
      </div>
    </div>
  )
}

export default function EvaluationsPage() {
  const [latestEval, setLatestEval] = useState<EvaluationLatestResponse | null>(null)
  const [runs, setRuns] = useState<EvaluationRunSummary[]>([])
  const [selectedRun, setSelectedRun] = useState<EvaluationRunDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'latest' | 'runs' | 'details' | 'ragas'>('latest')
  const [ragasSummary, setRagasSummary] = useState<RAGASSummary | null>(null)
  const authFetch = useAuthFetch()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [latestRes, runsRes, ragasRes] = await Promise.all([
        authFetch('/api/admin/evaluations/latest'),
        authFetch('/api/admin/evaluations/runs'),
        authFetch('/api/admin/evaluations/ragas-summary'),
      ])

      if (!latestRes.ok || !runsRes.ok) {
        throw new Error('Failed to fetch evaluation data')
      }

      const [latestData, runsData] = await Promise.all([
        latestRes.json(),
        runsRes.json(),
      ])

      setLatestEval(latestData)
      setRuns(runsData.runs)

      // RAGAS summary: 200 even when no report exists
      if (ragasRes.ok) {
        setRagasSummary(await ragasRes.json())
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const fetchRunDetails = async (runId: number) => {
    try {
      const res = await authFetch(`/api/admin/evaluations/${runId}`)
      if (!res.ok) {
        throw new Error('Failed to fetch run details')
      }
      const data = await res.json()
      setSelectedRun(data)
      setActiveTab('details')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString()
  }

  const formatLatency = (ms: number | null) => {
    if (ms === null) return 'N/A'
    return `${ms.toFixed(0)}ms`
  }

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 dark:bg-green-900/30 text-hiplink-success dark:text-green-400'
      case 'failed':
        return 'bg-red-100 dark:bg-red-900/30 text-hiplink-error dark:text-red-400'
      default:
        return 'bg-yellow-100 dark:bg-yellow-900/30 text-hiplink-warning dark:text-yellow-400'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-hiplink-secondary dark:text-dark-text-dim">Loading evaluation data...</div>
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
        <h2 className="text-xl font-bold text-hiplink-dark dark:text-dark-text">Evaluation Results</h2>
        <p className="text-hiplink-secondary dark:text-dark-text-dim mt-1">Track controlled RAG quality tests and failure reasons.</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 mb-6">
        {([
          { key: 'latest' as const, label: 'Latest Run', disabled: false },
          { key: 'runs' as const, label: 'All Runs', disabled: false },
          { key: 'details' as const, label: 'Run Details', disabled: !selectedRun },
          { key: 'ragas' as const, label: 'RAGAS Scores', disabled: false },
        ]).map(({ key, label, disabled }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            disabled={disabled}
            className={`px-4 py-2.5 rounded-lg font-medium transition-colors ${
              activeTab === key
                ? 'bg-hiplink-blue dark:bg-sky-600 text-white'
                : 'bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text border border-hiplink-border dark:border-dark-border hover:bg-gray-50 dark:hover:bg-dark-elevated disabled:opacity-50 disabled:cursor-not-allowed'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Latest Run Dashboard */}
      {activeTab === 'latest' && latestEval && (
        <div>
          {latestEval.latest_run ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
                <MetricCard label="Total Tests" value={latestEval.latest_run.total_tests} />
                <MetricCard label="Passed" value={latestEval.latest_run.passed_tests} highlight="success" />
                <MetricCard label="Failed" value={latestEval.latest_run.failed_tests} highlight="error" />
                <MetricCard 
                  label="Pass Rate" 
                  value={`${latestEval.pass_percentage.toFixed(1)}%`} 
                  highlight={latestEval.pass_percentage >= 70 ? 'success' : 'error'} 
                />
                <MetricCard label="Avg Latency" value={formatLatency(latestEval.average_latency_ms)} />
                <MetricCard label="Avg Top Score" value={latestEval.average_top_score?.toFixed(3) || 'N/A'} />
              </div>

              <div className="flex items-center gap-3 text-sm text-hiplink-secondary dark:text-dark-text-dim mb-4">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Last run: {latestEval.timestamp ? formatDate(latestEval.timestamp) : 'Never'}
                {latestEval.latest_run.status && (
                  <>
                    <span className="text-gray-300 dark:text-slate-600">•</span>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusBadgeClass(latestEval.latest_run.status)}`}>
                      {latestEval.latest_run.status}
                    </span>
                  </>
                )}
              </div>

              {/* Failed Cases */}
              {latestEval.failed_cases.length > 0 && (
                <div className="card dark:bg-dark-card overflow-hidden">
                  <div className="px-4 py-3 bg-red-50 dark:bg-red-900/20 border-b border-hiplink-border dark:border-dark-border">
                    <h3 className="font-semibold text-hiplink-error dark:text-red-400 flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      Failed Test Cases ({latestEval.failed_cases.length})
                    </h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-hiplink-border dark:divide-dark-border">
                      <thead className="bg-hiplink-background dark:bg-dark-elevated">
                        <tr>
                          <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Test ID</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Question</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Failure Reasons</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Missing Keywords</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Forbidden Found</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Expected Source</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Actual Sources</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                        {latestEval.failed_cases.map((tc) => (
                          <tr key={tc.test_case_id} className="hover:bg-hiplink-background dark:hover:bg-dark-elevated transition-colors">
                            <td className="px-4 py-3 text-xs font-mono text-hiplink-blue dark:text-sky-400">{tc.test_case_id}</td>
                            <td className="px-4 py-3 text-xs text-hiplink-dark dark:text-dark-text max-w-xs truncate">{tc.question}</td>
                            <td className="px-4 py-3 text-xs">
                              <div className="flex flex-wrap gap-1">
                                {tc.failure_reasons?.map((reason, i) => (
                                  <span key={i} className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-hiplink-error dark:text-red-400 rounded text-xs font-medium">
                                    {reason}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-4 py-3 text-xs">
                              <div className="flex flex-wrap gap-1">
                                {tc.missing_keywords?.map((kw, i) => (
                                  <span key={i} className="px-2 py-1 bg-amber-100 dark:bg-amber-900/30 text-hiplink-warning dark:text-amber-400 rounded text-xs">
                                    {kw}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-4 py-3 text-xs">
                              <div className="flex flex-wrap gap-1">
                                {tc.forbidden_keywords_found?.map((kw, i) => (
                                  <span key={i} className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded text-xs">
                                    {kw}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{tc.expected_source_file || 'N/A'}</td>
                            <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">
                              {tc.actual_source_files?.join(', ') || 'None'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {latestEval.failed_cases.length === 0 && (
                <div className="card dark:bg-dark-card p-8 border-hiplink-success dark:border-green-500 bg-green-50 dark:bg-green-900/20">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-hiplink-success dark:bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-lg font-semibold text-hiplink-success dark:text-green-400">All test cases passed!</p>
                      <p className="text-green-700 dark:text-green-300">Your RAG system is performing optimally.</p>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card dark:bg-dark-card p-8 text-center">
              <div className="w-16 h-16 bg-gray-100 dark:bg-dark-elevated rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-400 dark:text-dark-text-dim" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <p className="text-hiplink-secondary dark:text-dark-text-dim mb-4">No evaluation runs yet. Run evaluations using:</p>
              <code className="bg-hiplink-background dark:bg-dark-elevated dark:text-dark-text px-4 py-2 rounded text-sm font-mono">
                docker compose exec backend python /app/scripts/run_rag_evaluation.py
              </code>
            </div>
          )}
        </div>
      )}

      {/* All Runs Table */}
      {activeTab === 'runs' && (
        <div className="card dark:bg-dark-card overflow-hidden">
          <table className="min-w-full divide-y divide-hiplink-border dark:divide-dark-border">
            <thead className="bg-hiplink-background dark:bg-dark-elevated">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Run ID</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Timestamp</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Total</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Passed</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Failed</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Pass %</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Avg Latency</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
              {runs.map((run) => (
                <tr key={run.run_id} className="hover:bg-hiplink-background dark:hover:bg-dark-elevated transition-colors">
                  <td className="px-4 py-3 text-xs font-mono text-hiplink-blue dark:text-sky-400">{run.run_id}</td>
                  <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim whitespace-nowrap">{formatDate(run.created_at)}</td>
                  <td className="px-4 py-3 text-xs text-hiplink-dark dark:text-dark-text">{run.total_tests}</td>
                  <td className="px-4 py-3 text-xs text-hiplink-success dark:text-green-400">{run.passed_tests}</td>
                  <td className="px-4 py-3 text-xs text-hiplink-error dark:text-red-400">{run.failed_tests}</td>
                  <td className="px-4 py-3 text-xs">
                    <span className={run.pass_percentage >= 70 ? 'text-hiplink-success dark:text-green-400 font-medium' : 'text-hiplink-error dark:text-red-400 font-medium'}>
                      {run.pass_percentage.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{formatLatency(run.average_latency_ms)}</td>
                  <td className="px-4 py-3 text-xs">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadgeClass(run.status)}`}>
                      {run.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs">
                    <button
                      onClick={() => fetchRunDetails(run.run_id)}
                      className="text-hiplink-blue dark:text-sky-400 hover:text-hiplink-blue-dark dark:hover:text-sky-300 font-medium"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-4 py-8 text-center text-hiplink-secondary dark:text-dark-text-dim text-sm">No evaluation runs yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Run Details */}
      {activeTab === 'details' && selectedRun && (
        <div>
          {/* Summary */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            <MetricCard label="Total Tests" value={selectedRun.summary.total_tests} />
            <MetricCard label="Passed" value={selectedRun.summary.passed_tests} highlight="success" />
            <MetricCard label="Failed" value={selectedRun.summary.failed_tests} highlight="error" />
            <MetricCard 
              label="Pass Rate" 
              value={`${selectedRun.summary.pass_percentage.toFixed(1)}%`}
              highlight={selectedRun.summary.pass_percentage >= 70 ? 'success' : 'error'}
            />
            <MetricCard label="Avg Latency" value={formatLatency(selectedRun.summary.average_latency_ms)} />
            <MetricCard label="Avg Top Score" value={selectedRun.summary.average_top_score?.toFixed(3) || 'N/A'} />
          </div>

          <div className="flex items-center gap-3 text-sm text-hiplink-secondary dark:text-dark-text-dim mb-6">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Run {selectedRun.summary.run_id} - {formatDate(selectedRun.summary.created_at)}
          </div>

          {/* Failure Summary */}
          {selectedRun.failed_results.length > 0 && (
            <div className="card dark:bg-dark-card overflow-hidden mb-6">
              <div className="px-4 py-3 bg-red-50 dark:bg-red-900/20 border-b border-hiplink-border dark:border-dark-border">
                <h3 className="font-semibold text-hiplink-error dark:text-red-400">
                  Failure Summary ({selectedRun.failed_results.length} failed)
                </h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark dark:text-dark-text mb-2 flex items-center gap-2">
                      <span className="w-2 h-2 bg-amber-500 dark:bg-amber-400 rounded-full"></span>
                      Missing Keywords
                    </h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('missing_keywords')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue dark:text-sky-400">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary dark:text-dark-text-dim">{r.missing_keywords?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark dark:text-dark-text mb-2 flex items-center gap-2">
                      <span className="w-2 h-2 bg-purple-500 dark:bg-purple-400 rounded-full"></span>
                      Forbidden Keywords Found
                    </h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('forbidden_keywords_found')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue dark:text-sky-400">{r.test_case_id}:</span>{' '}
                        <span className="text-purple-600 dark:text-purple-400">{r.forbidden_keywords_found?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark dark:text-dark-text mb-2 flex items-center gap-2">
                      <span className="w-2 h-2 bg-red-500 dark:bg-red-400 rounded-full"></span>
                      Wrong Source
                    </h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('wrong_source')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue dark:text-sky-400">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary dark:text-dark-text-dim">Expected: {r.expected_source_file}, Got: {r.actual_source_files?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark dark:text-dark-text mb-2 flex items-center gap-2">
                      <span className="w-2 h-2 bg-amber-500 dark:bg-amber-400 rounded-full"></span>
                      Missing Citations
                    </h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('missing_citations')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue dark:text-sky-400">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary dark:text-dark-text-dim">Expected citations but got {r.citation_count}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark dark:text-dark-text mb-2 flex items-center gap-2">
                      <span className="w-2 h-2 bg-red-500 dark:bg-red-400 rounded-full"></span>
                      Fallback Failures
                    </h4>
                    {selectedRun.failed_results.filter(r => 
                      r.failure_reasons?.includes('expected_fallback_but_answered') ||
                      r.failure_reasons?.includes('expected_answer_but_fallback')
                    ).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue dark:text-sky-400">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary dark:text-dark-text-dim">
                          {r.expected_fallback ? 'Expected fallback, got answer' : 'Expected answer, got fallback'}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark dark:text-dark-text mb-2 flex items-center gap-2">
                      <span className="w-2 h-2 bg-red-500 dark:bg-red-400 rounded-full"></span>
                      Blocked Incorrectly
                    </h4>
                    {selectedRun.failed_results.filter(r => 
                      r.failure_reasons?.includes('incorrectly_blocked') ||
                      r.failure_reasons?.includes('should_have_blocked')
                    ).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue dark:text-sky-400">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary dark:text-dark-text-dim">
                          {r.actual_blocked ? 'Should have answered' : 'Should have blocked'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* All Test Results */}
          <div className="card dark:bg-dark-card overflow-hidden">
            <div className="px-4 py-3 bg-hiplink-background dark:bg-dark-elevated border-b border-hiplink-border dark:border-dark-border">
              <h3 className="font-semibold text-hiplink-dark dark:text-dark-text">All Test Results</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-hiplink-border dark:divide-dark-border">
                <thead className="bg-hiplink-background dark:bg-dark-elevated">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Status</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Test ID</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Question</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Expected Source</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Citations</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Top Score</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-hiplink-secondary dark:text-dark-text-dim uppercase tracking-wider">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-hiplink-border dark:divide-dark-border">
                  {selectedRun.results.map((result) => (
                    <tr key={result.test_case_id} className={`hover:bg-hiplink-background dark:hover:bg-dark-elevated transition-colors ${!result.passed ? 'bg-red-50 dark:bg-red-900/10' : ''}`}>
                      <td className="px-4 py-3 text-xs">
                        {result.passed ? (
                          <span className="text-hiplink-success dark:text-green-400 font-bold">PASS</span>
                        ) : (
                          <span className="text-hiplink-error dark:text-red-400 font-bold">FAIL</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-xs font-mono text-hiplink-blue dark:text-sky-400">{result.test_case_id}</td>
                      <td className="px-4 py-3 text-xs text-hiplink-dark dark:text-dark-text max-w-xs truncate">{result.question}</td>
                      <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{result.expected_source_file || 'N/A'}</td>
                      <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{result.citation_count}</td>
                      <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{result.top_score?.toFixed(3) || 'N/A'}</td>
                      <td className="px-4 py-3 text-xs text-hiplink-secondary dark:text-dark-text-dim">{formatLatency(result.latency_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* RAGAS Scores Tab */}
      {activeTab === 'ragas' && (
        <div>
          <div className="mb-4 flex items-center gap-2">
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
              ragasSummary?.enabled
                ? 'bg-green-100 dark:bg-green-900/30 text-hiplink-success dark:text-green-400'
                : 'bg-gray-100 dark:bg-dark-elevated text-hiplink-secondary dark:text-dark-text-muted'
            }`}>
              {ragasSummary?.enabled ? 'Enabled' : 'Disabled'}
            </span>
            <span className="text-xs text-hiplink-secondary dark:text-dark-text-dim">
              RAGAS is an additive quality layer — custom evaluation (20/20) remains the regression gate.
            </span>
          </div>

          {/* Status Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard
              label="RAGAS Status"
              value={!ragasSummary ? 'Loading…' : ragasSummary.available ? 'Available' : 'Not Installed'}
              highlight={!ragasSummary ? 'neutral' : ragasSummary.available ? 'success' : 'warning'}
            />
            <MetricCard
              label="Evaluator"
              value={ragasSummary?.evaluator_provider ? `${ragasSummary.evaluator_provider} / ${ragasSummary.evaluator_model}` : '—'}
            />
            <MetricCard
              label="Latest Report"
              value={!ragasSummary ? '—' : ragasSummary.latest_report_found ? (ragasSummary.latest_report_name || 'Found') : 'None'}
              highlight={!ragasSummary ? 'neutral' : ragasSummary.latest_report_found ? 'success' : 'warning'}
            />
            <MetricCard
              label="Last Run"
              value={
                !ragasSummary?.latest_timestamp
                  ? 'Never'
                  : new Date(ragasSummary.latest_timestamp).toLocaleDateString()
              }
            />
          </div>

          {/* RAGAS Test Parameters Section */}
          <div className="mb-6">
            <h3 className="text-base font-semibold text-hiplink-dark dark:text-dark-text mb-4">
              RAGAS Test Parameters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <RAGASMetricCard
                name="Faithfulness"
                purpose="Checks whether the answer is grounded in the retrieved context."
                score={ragasSummary?.metrics?.faithfulness ?? null}
                threshold={ragasSummary?.threshold_faithfulness ?? 0.5}
                skipped={false}
              />
              <RAGASMetricCard
                name="Answer Relevancy"
                purpose="Checks whether the answer directly addresses the user question."
                score={ragasSummary?.metrics?.answer_relevancy ?? null}
                threshold={ragasSummary?.threshold_answer_relevancy ?? 0.5}
                skipped={false}
              />
              <RAGASMetricCard
                name="Context Precision"
                purpose="Checks whether retrieved context is relevant and useful."
                score={ragasSummary?.metrics?.context_precision ?? null}
                threshold={ragasSummary?.threshold_context_precision ?? 0.5}
                skipped={false}
              />
              <RAGASMetricCard
                name="Context Recall"
                purpose="Checks whether all required supporting context was retrieved."
                score={ragasSummary?.metrics?.context_recall ?? null}
                threshold={0.5}
                skipped={true}
                skippedReason="ground_truth required"
              />
              <RAGASMetricCard
                name="Answer Correctness"
                purpose="Checks final answer correctness against ground_truth."
                score={ragasSummary?.metrics?.answer_correctness ?? null}
                threshold={0.5}
                skipped={true}
                skippedReason="ground_truth required"
              />
            </div>
          </div>

          {/* Thresholds Section */}
          <div className="mb-6 card dark:bg-dark-card p-4">
            <h3 className="text-sm font-semibold text-hiplink-dark dark:text-dark-text mb-3">
              Thresholds
            </h3>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-1">Faithfulness</div>
                <div className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">
                  {ragasSummary?.threshold_faithfulness != null ? ragasSummary.threshold_faithfulness : 'Not configured'}
                </div>
              </div>
              <div>
                <div className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-1">Answer Relevancy</div>
                <div className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">
                  {ragasSummary?.threshold_answer_relevancy != null ? ragasSummary.threshold_answer_relevancy : 'Not configured'}
                </div>
              </div>
              <div>
                <div className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-1">Context Precision</div>
                <div className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">
                  {ragasSummary?.threshold_context_precision != null ? ragasSummary.threshold_context_precision : 'Not configured'}
                </div>
              </div>
            </div>
          </div>

          {/* Warnings */}
          {ragasSummary?.warnings && ragasSummary.warnings.length > 0 && (
            <div className="mb-6 card dark:bg-dark-card p-4 border border-yellow-200 dark:border-yellow-800">
              <h3 className="text-sm font-semibold text-yellow-700 dark:text-yellow-400 mb-2">Notes</h3>
              <ul className="space-y-1">
                {ragasSummary.warnings.map((w, i) => (
                  <li key={i} className="text-xs text-yellow-700 dark:text-yellow-300 flex items-start gap-2">
                    <span className="mt-0.5">•</span>
                    <span>{w}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* No Report Yet - Improved messaging */}
          {ragasSummary && !ragasSummary.latest_report_found && (
            <div className="card dark:bg-dark-card p-6 text-center border border-blue-200 dark:border-blue-800">
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
                <svg className="w-7 h-7 text-blue-500 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <p className="text-sm font-medium text-hiplink-secondary dark:text-dark-text mb-1">
                No RAGAS report generated yet.
              </p>
              <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-4 max-w-md mx-auto">
                RAGAS is available, but only dry-run validation has been performed. Run RAGAS without --dry-run to generate actual metric scores.
              </p>
              <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim">
                Custom evaluation ({latestEval?.latest_run?.total_tests || 20}/{latestEval?.latest_run?.passed_tests || 20}) remains the primary regression gate.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}