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

  if (loading) {
    return <div className="text-center text-gray-500 py-8">Loading evaluation data...</div>
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
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Evaluation Results</h2>

      {/* Tab Navigation */}
      <div className="flex space-x-2 mb-6">
        <button
          onClick={() => setActiveTab('latest')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'latest'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Latest Run
        </button>
        <button
          onClick={() => setActiveTab('runs')}
          className={`px-4 py-2 rounded font-medium ${
            activeTab === 'runs'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          All Runs
        </button>
        {selectedRun && (
          <button
            onClick={() => setActiveTab('details')}
            className={`px-4 py-2 rounded font-medium ${
              activeTab === 'details'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Run Details
          </button>
        )}
      </div>

      {/* Latest Run Dashboard */}
      {activeTab === 'latest' && latestEval && (
        <div>
          {latestEval.latest_run ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="text-2xl font-bold text-gray-900">
                    {latestEval.latest_run.total_tests}
                  </div>
                  <div className="text-sm text-gray-500">Total Tests</div>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-600">
                    {latestEval.latest_run.passed_tests}
                  </div>
                  <div className="text-sm text-gray-500">Passed</div>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="text-2xl font-bold text-red-600">
                    {latestEval.latest_run.failed_tests}
                  </div>
                  <div className="text-sm text-gray-500">Failed</div>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className={`text-2xl font-bold ${
                    latestEval.pass_percentage >= 70 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {latestEval.pass_percentage.toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-500">Pass Rate</div>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="text-2xl font-bold text-gray-900">
                    {formatLatency(latestEval.average_latency_ms)}
                  </div>
                  <div className="text-sm text-gray-500">Avg Latency</div>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="text-2xl font-bold text-gray-900">
                    {latestEval.average_top_score?.toFixed(3) || 'N/A'}
                  </div>
                  <div className="text-sm text-gray-500">Avg Top Score</div>
                </div>
              </div>

              <div className="text-sm text-gray-500 mb-4">
                Last run: {latestEval.timestamp ? formatDate(latestEval.timestamp) : 'Never'}
              </div>

              {/* Failed Cases */}
              {latestEval.failed_cases.length > 0 && (
                <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                  <div className="px-4 py-3 bg-red-50 border-b border-gray-200">
                    <h3 className="font-semibold text-red-700">
                      Failed Test Cases ({latestEval.failed_cases.length})
                    </h3>
                  </div>
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Test ID</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Question</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Failure Reasons</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Missing Keywords</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Forbidden Found</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Expected Source</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Actual Sources</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {latestEval.failed_cases.map((tc) => (
                        <tr key={tc.test_case_id} className="hover:bg-gray-50">
                          <td className="px-3 py-2 text-xs font-mono text-blue-600">{tc.test_case_id}</td>
                          <td className="px-3 py-2 text-xs text-gray-900 max-w-xs truncate">{tc.question}</td>
                          <td className="px-3 py-2 text-xs">
                            <div className="flex flex-wrap gap-1">
                              {tc.failure_reasons?.map((reason, i) => (
                                <span key={i} className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs">
                                  {reason}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-3 py-2 text-xs">
                            <div className="flex flex-wrap gap-1">
                              {tc.missing_keywords?.map((kw, i) => (
                                <span key={i} className="px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded text-xs">
                                  {kw}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-3 py-2 text-xs">
                            <div className="flex flex-wrap gap-1">
                              {tc.forbidden_keywords_found?.map((kw, i) => (
                                <span key={i} className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs">
                                  {kw}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-3 py-2 text-xs text-gray-600">{tc.expected_source_file || 'N/A'}</td>
                          <td className="px-3 py-2 text-xs text-gray-600">
                            {tc.actual_source_files?.join(', ') || 'None'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {latestEval.failed_cases.length === 0 && (
                <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
                  All test cases passed! 🎉
                </div>
              )}
            </>
          ) : (
            <div className="bg-gray-100 border border-gray-300 text-gray-700 px-4 py-3 rounded">
              No evaluation runs yet. Run evaluations using:<br/>
              <code className="bg-gray-200 px-2 py-1 rounded mt-2 inline-block">
                docker compose exec backend python /app/scripts/run_rag_evaluation.py
              </code>
            </div>
          )}
        </div>
      )}

      {/* All Runs Table */}
      {activeTab === 'runs' && (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Run ID</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Timestamp</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Total</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Passed</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Failed</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Pass %</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Avg Latency</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Status</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {runs.map((run) => (
                <tr key={run.run_id} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-xs font-mono text-blue-600">{run.run_id}</td>
                  <td className="px-3 py-2 text-xs text-gray-500 whitespace-nowrap">{formatDate(run.created_at)}</td>
                  <td className="px-3 py-2 text-xs text-gray-900">{run.total_tests}</td>
                  <td className="px-3 py-2 text-xs text-green-600">{run.passed_tests}</td>
                  <td className="px-3 py-2 text-xs text-red-600">{run.failed_tests}</td>
                  <td className="px-3 py-2 text-xs">
                    <span className={run.pass_percentage >= 70 ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
                      {run.pass_percentage.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-600">{formatLatency(run.average_latency_ms)}</td>
                  <td className="px-3 py-2 text-xs">
                    <span className={`px-2 py-0.5 rounded-full text-xs ${
                      run.status === 'completed' ? 'bg-green-100 text-green-700' :
                      run.status === 'failed' ? 'bg-red-100 text-red-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {run.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs">
                    <button
                      onClick={() => fetchRunDetails(run.run_id)}
                      className="text-blue-600 hover:text-blue-800 font-medium"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-3 py-4 text-center text-gray-500 text-sm">No evaluation runs yet.</td>
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
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-gray-900">
                {selectedRun.summary.total_tests}
              </div>
              <div className="text-sm text-gray-500">Total Tests</div>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-600">
                {selectedRun.summary.passed_tests}
              </div>
              <div className="text-sm text-gray-500">Passed</div>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-red-600">
                {selectedRun.summary.failed_tests}
              </div>
              <div className="text-sm text-gray-500">Failed</div>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className={`text-2xl font-bold ${
                selectedRun.summary.pass_percentage >= 70 ? 'text-green-600' : 'text-red-600'
              }`}>
                {selectedRun.summary.pass_percentage.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">Pass Rate</div>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-gray-900">
                {formatLatency(selectedRun.summary.average_latency_ms)}
              </div>
              <div className="text-sm text-gray-500">Avg Latency</div>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-2xl font-bold text-gray-900">
                {selectedRun.summary.average_top_score?.toFixed(3) || 'N/A'}
              </div>
              <div className="text-sm text-gray-500">Avg Top Score</div>
            </div>
          </div>

          <div className="text-sm text-gray-500 mb-4">
            Run {selectedRun.summary.run_id} - {formatDate(selectedRun.summary.created_at)}
          </div>

          {/* Failure Summary */}
          {selectedRun.failed_results.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg overflow-hidden mb-6">
              <div className="px-4 py-3 bg-red-50 border-b border-gray-200">
                <h3 className="font-semibold text-red-700">
                  Failure Summary ({selectedRun.failed_results.length} failed)
                </h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Missing Keywords</h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('missing_keywords')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-blue-600">{r.test_case_id}:</span>{' '}
                        <span className="text-gray-600">{r.missing_keywords?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Forbidden Keywords Found</h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('forbidden_keywords_found')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-blue-600">{r.test_case_id}:</span>{' '}
                        <span className="text-purple-600">{r.forbidden_keywords_found?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Wrong Source</h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('wrong_source')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-blue-600">{r.test_case_id}:</span>{' '}
                        <span className="text-gray-600">Expected: {r.expected_source_file}, Got: {r.actual_source_files?.join(', ')}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Missing Citations</h4>
                    {selectedRun.failed_results.filter(r => r.failure_reasons?.includes('missing_citations')).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-blue-600">{r.test_case_id}:</span>{' '}
                        <span className="text-gray-600">Expected citations but got {r.citation_count}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Fallback Failures</h4>
                    {selectedRun.failed_results.filter(r => 
                      r.failure_reasons?.includes('expected_fallback_but_answered') ||
                      r.failure_reasons?.includes('expected_answer_but_fallback')
                    ).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-blue-600">{r.test_case_id}:</span>{' '}
                        <span className="text-gray-600">
                          {r.expected_fallback ? 'Expected fallback, got answer' : 'Expected answer, got fallback'}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Blocked Incorrectly</h4>
                    {selectedRun.failed_results.filter(r => 
                      r.failure_reasons?.includes('incorrectly_blocked') ||
                      r.failure_reasons?.includes('should_have_blocked')
                    ).map(r => (
                      <div key={r.test_case_id} className="text-xs mb-1">
                        <span className="font-mono text-blue-600">{r.test_case_id}:</span>{' '}
                        <span className="text-gray-600">
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
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
              <h3 className="font-semibold text-gray-700">All Test Results</h3>
            </div>
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Status</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Test ID</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Question</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Expected Source</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Citations</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Top Score</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {selectedRun.results.map((result) => (
                  <tr key={result.test_case_id} className={`hover:bg-gray-50 ${!result.passed ? 'bg-red-50' : ''}`}>
                    <td className="px-3 py-2 text-xs">
                      {result.passed ? (
                        <span className="text-green-600 font-medium">PASS</span>
                      ) : (
                        <span className="text-red-600 font-medium">FAIL</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-xs font-mono text-blue-600">{result.test_case_id}</td>
                    <td className="px-3 py-2 text-xs text-gray-900 max-w-xs truncate">{result.question}</td>
                    <td className="px-3 py-2 text-xs text-gray-600">{result.expected_source_file || 'N/A'}</td>
                    <td className="px-3 py-2 text-xs text-gray-600">{result.citation_count}</td>
                    <td className="px-3 py-2 text-xs text-gray-600">{result.top_score?.toFixed(3) || 'N/A'}</td>
                    <td className="px-3 py-2 text-xs text-gray-600">{formatLatency(result.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}