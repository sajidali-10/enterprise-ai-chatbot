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

interface EvaluationRunsListResponse {
  runs: EvaluationRunSummary[]
  total: number
}

interface MetricCardProps {
  label: string
  value: string | number
  highlight?: 'success' | 'error' | 'warning' | 'blue'
}

function MetricCard({ label, value, highlight }: MetricCardProps) {
  const colorClasses = {
    success: 'text-hiplink-success',
    error: 'text-hiplink-error',
    warning: 'text-hiplink-warning',
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

export default function EvaluationsPage() {
  const [latestEval, setLatestEval] = useState<EvaluationLatestResponse | null>(null)
  const [runs, setRuns] = useState<EvaluationRunSummary[]>([])
  const [selectedRun, setSelectedRun] = useState<EvaluationRunDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'latest' | 'runs' | 'details'>('latest')
  const authFetch = useAuthFetch()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [latestRes, runsRes] = await Promise.all([
        authFetch('/api/admin/evaluations/latest'),
        authFetch('/api/admin/evaluations/runs'),
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
        return 'bg-green-100 text-hiplink-success'
      case 'failed':
        return 'bg-red-100 text-hiplink-error'
      default:
        return 'bg-yellow-100 text-hiplink-warning'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-hiplink-secondary">Loading evaluation data...</div>
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
      <h2 className="text-xl font-semibold text-hiplink-dark mb-6">Evaluation Results</h2>

      {/* Tab Navigation */}
      <div className="flex space-x-2 mb-6">
        {(['latest', 'runs', 'details'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            disabled={tab === 'details' && !selectedRun}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === tab
                ? 'bg-hiplink-blue text-white'
                : 'bg-white text-hiplink-dark border border-hiplink-border hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed'
            }`}
          >
            {tab === 'latest' ? 'Latest Run' : tab === 'runs' ? 'All Runs' : 'Run Details'}
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

              <div className="text-sm text-hiplink-secondary mb-4">
                Last run: {latestEval.timestamp ? formatDate(latestEval.timestamp) : 'Never'}
              </div>

              {/* Failed Cases */}
              {latestEval.failed_cases.length > 0 && (
                <div className="card overflow-hidden">
                  <div className="px-4 py-3 bg-red-50 border-b border-hiplink-border">
                    <h3 className="font-semibold text-hiplink-error">
                      Failed Test Cases ({latestEval.failed_cases.length})
                    </h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-hiplink-border">
                      <thead className="bg-hiplink-background">
                        <tr>
                          <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Test ID</th>
                          <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Question</th>
                          <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Failure Reasons</th>
                          <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Missing Keywords</th>
                          <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Forbidden Found</th>
                          <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Expected Source</th>
                          <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Actual Sources</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-hiplink-border">
                        {latestEval.failed_cases.map((tc) => (
                          <tr key={tc.test_case_id} className="hover:bg-hiplink-background transition-colors">
                            <td className="px-3 py-3 text-xs font-mono text-hiplink-blue">{tc.test_case_id}</td>
                            <td className="px-3 py-3 text-xs text-hiplink-dark max-w-xs truncate">{tc.question}</td>
                            <td className="px-3 py-3 text-xs">
                              <div className="flex flex-wrap gap-1">
                                {tc.failure_reasons?.map((reason, i) => (
                                  <span key={i} className="px-2 py-0.5 bg-red-100 text-hiplink-error rounded text-xs font-medium">
                                    {reason}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-3 py-3 text-xs">
                              <div className="flex flex-wrap gap-1">
                                {tc.missing_keywords?.map((kw, i) => (
                                  <span key={i} className="px-2 py-0.5 bg-yellow-100 text-hiplink-warning rounded text-xs">
                                    {kw}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-3 py-3 text-xs">
                              <div className="flex flex-wrap gap-1">
                                {tc.forbidden_keywords_found?.map((kw, i) => (
                                  <span key={i} className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs">
                                    {kw}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-3 py-3 text-xs text-hiplink-secondary">{tc.expected_source_file || 'N/A'}</td>
                            <td className="px-3 py-3 text-xs text-hiplink-secondary">
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
                <div className="card p-6 border-hiplink-success bg-green-50">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-hiplink-success rounded-full flex items-center justify-center">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-semibold text-hiplink-success">All test cases passed!</p>
                      <p className="text-sm text-green-700">🎉 Your RAG system is performing optimally.</p>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card p-6">
              <p className="text-hiplink-secondary mb-4">No evaluation runs yet. Run evaluations using:</p>
              <code className="bg-hiplink-background px-3 py-2 rounded text-sm font-mono">
                docker compose exec backend python /app/scripts/run_rag_evaluation.py
              </code>
            </div>
          )}
        </div>
      )}

      {/* All Runs Table */}
      {activeTab === 'runs' && (
        <div className="card overflow-hidden">
          <table className="min-w-full divide-y divide-hiplink-border">
            <thead className="bg-hiplink-background">
              <tr>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Run ID</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Timestamp</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Total</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Passed</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Failed</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Pass %</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Avg Latency</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Status</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hiplink-border">
              {runs.map((run) => (
                <tr key={run.run_id} className="hover:bg-hiplink-background transition-colors">
                  <td className="px-3 py-3 text-xs font-mono text-hiplink-blue">{run.run_id}</td>
                  <td className="px-3 py-3 text-xs text-hiplink-secondary whitespace-nowrap">{formatDate(run.created_at)}</td>
                  <td className="px-3 py-3 text-xs text-hiplink-dark">{run.total_tests}</td>
                  <td className="px-3 py-3 text-xs text-hiplink-success">{run.passed_tests}</td>
                  <td className="px-3 py-3 text-xs text-hiplink-error">{run.failed_tests}</td>
                  <td className="px-3 py-3 text-xs">
                    <span className={run.pass_percentage >= 70 ? 'text-hiplink-success font-medium' : 'text-hiplink-error font-medium'}>
                      {run.pass_percentage.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-3 py-3 text-xs text-hiplink-secondary">{formatLatency(run.average_latency_ms)}</td>
                  <td className="px-3 py-3 text-xs">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusBadgeClass(run.status)}`}>
                      {run.status}
                    </span>
                  </td>
                  <td className="px-3 py-3 text-xs">
                    <button
                      onClick={() => fetchRunDetails(run.run_id)}
                      className="text-hiplink-blue hover:text-hiplink-blue-dark font-medium"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-3 py-8 text-center text-hiplink-secondary text-sm">No evaluation runs yet.</td>
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

          <div className="text-sm text-hiplink-secondary mb-4">
            Run {selectedRun.summary.run_id} - {formatDate(selectedRun.summary.created_at)}
          </div>

          {/* Failure Summary */}
          {selectedRun.failed_results.length > 0 && (
            <div className="card overflow-hidden mb-6">
              <div className="px-4 py-3 bg-red-50 border-b border-hiplink-border">
                <h3 className="font-semibold text-hiplink-error">
                  Failure Summary ({selectedRun.failed_results.length} failed)
                </h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark mb-2">Missing Keywords</h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('missing_keywords')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary">{r.missing_keywords?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark mb-2">Forbidden Keywords Found</h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('forbidden_keywords_found')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue">{r.test_case_id}:</span>{' '}
                        <span className="text-purple-600">{r.forbidden_keywords_found?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark mb-2">Wrong Source</h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('wrong_source')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary">Expected: {r.expected_source_file}, Got: {r.actual_source_files?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark mb-2">Missing Citations</h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('missing_citations')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary">Expected citations but got {r.citation_count}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark mb-2">Fallback Failures</h4>
                    {selectedRun.failed_results.filter(r => 
                      r.failure_reasons?.includes('expected_fallback_but_answered') ||
                      r.failure_reasons?.includes('expected_answer_but_fallback')
                    ).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary">
                          {r.expected_fallback ? 'Expected fallback, got answer' : 'Expected answer, got fallback'}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-hiplink-dark mb-2">Blocked Incorrectly</h4>
                    {selectedRun.failed_results.filter(r => 
                      r.failure_reasons?.includes('incorrectly_blocked') ||
                      r.failure_reasons?.includes('should_have_blocked')
                    ).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-hiplink-blue">{r.test_case_id}:</span>{' '}
                        <span className="text-hiplink-secondary">
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
          <div className="card overflow-hidden">
            <div className="px-4 py-3 bg-hiplink-background border-b border-hiplink-border">
              <h3 className="font-semibold text-hiplink-dark">All Test Results</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-hiplink-border">
                <thead className="bg-hiplink-background">
                  <tr>
                    <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Status</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Test ID</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Question</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Expected Source</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Citations</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Top Score</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-hiplink-secondary uppercase tracking-wider">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-hiplink-border">
                  {selectedRun.results.map((result) => (
                    <tr key={result.test_case_id} className={`hover:bg-hiplink-background transition-colors ${!result.passed ? 'bg-red-50' : ''}`}>
                      <td className="px-3 py-3 text-xs">
                        {result.passed ? (
                          <span className="text-hiplink-success font-bold">PASS</span>
                        ) : (
                          <span className="text-hiplink-error font-bold">FAIL</span>
                        )}
                      </td>
                      <td className="px-3 py-3 text-xs font-mono text-hiplink-blue">{result.test_case_id}</td>
                      <td className="px-3 py-3 text-xs text-hiplink-dark max-w-xs truncate">{result.question}</td>
                      <td className="px-3 py-3 text-xs text-hiplink-secondary">{result.expected_source_file || 'N/A'}</td>
                      <td className="px-3 py-3 text-xs text-hiplink-secondary">{result.citation_count}</td>
                      <td className="px-3 py-3 text-xs text-hiplink-secondary">{result.top_score?.toFixed(3) || 'N/A'}</td>
                      <td className="px-3 py-3 text-xs text-hiplink-secondary">{formatLatency(result.latency_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}