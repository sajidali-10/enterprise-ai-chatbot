'use client'

import { useEffect, useState, useCallback } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { getApiBaseUrl } from '@/lib/api'
import ProtectedRoute from '@/components/ProtectedRoute'

interface AuditLogEntry {
  id: number
  username: string
  action: string
  request_ip: string | null
  status: string
  error_message: string | null
  details: any
  created_at: string | null
}

interface AuditLogResponse {
  entries: AuditLogEntry[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export default function AuditLogsPage() {
  const { auth } = useAuth()
  const [logs, setLogs] = useState<AuditLogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [pageSize] = useState(20)
  const [totalPages, setTotalPages] = useState(1)
  const [total, setTotal] = useState(0)

  // Filters
  const [actionFilter, setActionFilter] = useState('')
  const [usernameFilter, setUsernameFilter] = useState('')
  const [resultFilter, setResultFilter] = useState('')

  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null

  const fetchLogs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams()
      params.set('page', String(page))
      params.set('page_size', String(pageSize))
      if (actionFilter) params.set('action', actionFilter)
      if (usernameFilter) params.set('username', usernameFilter)
      if (resultFilter) params.set('result', resultFilter)

      const res = await fetch(`${getApiBaseUrl()}/api/admin/audit-logs?${params.toString()}`, {
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      const data: AuditLogResponse = await res.json()
      setLogs(data.entries)
      setTotalPages(data.total_pages)
      setTotal(data.total)
    } catch (err: any) {
      setError(err.message || 'Failed to load audit logs')
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, actionFilter, usernameFilter, resultFilter, token])

  useEffect(() => {
    if (auth?.is_admin) {
      fetchLogs()
    }
  }, [fetchLogs, auth?.is_admin])

  if (!auth?.is_admin) {
    return (
      <ProtectedRoute>
        <div className="min-h-screen bg-hiplink-background dark:bg-dark-bg flex items-center justify-center p-4">
          <div className="card dark:bg-dark-card p-8 text-center max-w-md">
            <h1 className="text-xl font-bold text-hiplink-dark dark:text-dark-text mb-2">Access Denied</h1>
            <p className="text-hiplink-secondary dark:text-dark-text-dim">Admin access required to view audit logs.</p>
          </div>
        </div>
      </ProtectedRoute>
    )
  }

  return (
    <ProtectedRoute requirePermission="canAccessObservability">
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-hiplink-dark dark:text-dark-text">Audit Logs</h1>
          <span className="text-sm text-hiplink-secondary dark:text-dark-text-dim">
            {total} total entries
          </span>
        </div>

        {/* Filters */}
        <div className="card dark:bg-dark-card p-4 mb-4">
          <div className="flex flex-wrap gap-3">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs font-medium text-hiplink-dark dark:text-dark-text mb-1">Action</label>
              <input
                type="text"
                value={actionFilter}
                onChange={(e) => { setActionFilter(e.target.value); setPage(1); }}
                placeholder="e.g. login_success"
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text text-sm"
              />
            </div>
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs font-medium text-hiplink-dark dark:text-dark-text mb-1">Username</label>
              <input
                type="text"
                value={usernameFilter}
                onChange={(e) => { setUsernameFilter(e.target.value); setPage(1); }}
                placeholder="Filter by username"
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text text-sm"
              />
            </div>
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs font-medium text-hiplink-dark dark:text-dark-text mb-1">Result</label>
              <select
                value={resultFilter}
                onChange={(e) => { setResultFilter(e.target.value); setPage(1); }}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text text-sm"
              >
                <option value="">All</option>
                <option value="success">Success</option>
                <option value="failure">Failure</option>
              </select>
            </div>
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm mb-4">
            {error}
          </div>
        )}

        {/* Table */}
        <div className="card dark:bg-dark-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-dark-elevated text-left">
                <tr>
                  <th className="px-4 py-3 font-medium text-hiplink-dark dark:text-dark-text">ID</th>
                  <th className="px-4 py-3 font-medium text-hiplink-dark dark:text-dark-text">Timestamp</th>
                  <th className="px-4 py-3 font-medium text-hiplink-dark dark:text-dark-text">Action</th>
                  <th className="px-4 py-3 font-medium text-hiplink-dark dark:text-dark-text">User</th>
                  <th className="px-4 py-3 font-medium text-hiplink-dark dark:text-dark-text">IP</th>
                  <th className="px-4 py-3 font-medium text-hiplink-dark dark:text-dark-text">Result</th>
                  <th className="px-4 py-3 font-medium text-hiplink-dark dark:text-dark-text">Details</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-hiplink-secondary dark:text-dark-text-dim">
                      Loading...
                    </td>
                  </tr>
                ) : logs.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-hiplink-secondary dark:text-dark-text-dim">
                      No audit logs found.
                    </td>
                  </tr>
                ) : (
                  logs.map((log) => (
                    <tr key={log.id} className="border-t border-hiplink-border dark:border-dark-border hover:bg-gray-50 dark:hover:bg-dark-elevated">
                      <td className="px-4 py-3 text-hiplink-dark dark:text-dark-text">{log.id}</td>
                      <td className="px-4 py-3 text-hiplink-secondary dark:text-dark-text-dim">
                        {log.created_at ? new Date(log.created_at).toLocaleString() : '-'}
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex px-2 py-0.5 rounded text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">
                          {log.action}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-hiplink-dark dark:text-dark-text">{log.username}</td>
                      <td className="px-4 py-3 text-hiplink-secondary dark:text-dark-text-dim font-mono text-xs">{log.request_ip || '-'}</td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex px-2 py-0.5 rounded text-xs font-medium ${
                          log.status === 'success'
                            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                            : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                        }`}>
                          {log.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-hiplink-secondary dark:text-dark-text-dim text-xs max-w-[200px] truncate">
                        {log.details ? JSON.stringify(log.details) : log.error_message || '-'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-hiplink-border dark:border-dark-border">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="px-3 py-1.5 rounded-lg text-sm bg-gray-100 dark:bg-dark-elevated text-hiplink-secondary dark:text-dark-text-muted hover:bg-gray-200 dark:hover:bg-dark-border transition-colors disabled:opacity-50"
              >
                &larr; Previous
              </button>
              <span className="text-sm text-hiplink-secondary dark:text-dark-text-dim">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="px-3 py-1.5 rounded-lg text-sm bg-gray-100 dark:bg-dark-elevated text-hiplink-secondary dark:text-dark-text-muted hover:bg-gray-200 dark:hover:bg-dark-border transition-colors disabled:opacity-50"
              >
                Next &rarr;
              </button>
            </div>
          )}
        </div>
      </div>
    </ProtectedRoute>
  )
}
