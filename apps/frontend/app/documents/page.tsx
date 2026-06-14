'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useAuthFetch } from '@/hooks/useApi'
import { useAuth } from '@/contexts/AuthContext'
import { normalizePermissions, type UserRole, type PermissionFlags } from '@/lib/permissions'
import ProtectedRoute from '@/components/ProtectedRoute'

interface Document {
  id: string
  original_name: string
  mime_type: string
  size_bytes: number
  status: string
  created_at: string
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

export default function DocumentsPage() {
  return (
    <ProtectedRoute requirePermission="canViewDocuments">
      <DocumentsPageInner />
    </ProtectedRoute>
  )
}

function DocumentsPageInner() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const authFetch = useAuthFetch()
  const { devUser, auth } = useAuth()
  
  // Determine effective role and permissions
  const rawRole = auth?.role || 
    (devUser === 'admin_user' ? 'admin' : 
     devUser === 'regular_user' ? 'user' : 
     devUser === 'viewer_user' ? 'viewer' : undefined)
  const perms: PermissionFlags = normalizePermissions(auth?.permissions, rawRole)

  useEffect(() => {
    fetchDocuments()
  }, [])

  async function fetchDocuments() {
    setLoading(true)
    setError(null)
    try {
      const res = await authFetch('/api/documents')
      if (!res.ok) {
        throw new Error(`Failed to fetch documents: HTTP ${res.status}`)
      }
      const data = await res.json()
      setDocuments(data || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents')
    } finally {
      setLoading(false)
    }
  }

  async function handleReindexAll() {
    if (!confirm('This will re-index all documents. Continue?')) return
    
    setActionLoading('reindex')
    setActionMessage(null)
    try {
      const res = await authFetch('/api/documents/reindex-all', { method: 'POST' })
      if (!res.ok) {
        throw new Error(`Failed to reindex: HTTP ${res.status}`)
      }
      const data = await res.json()
      setActionMessage({ 
        type: 'success', 
        text: `Re-indexed ${data.reindexed} of ${data.total_documents} documents` 
      })
      fetchDocuments()
    } catch (err) {
      setActionMessage({ 
        type: 'error', 
        text: err instanceof Error ? err.message : 'Failed to reindex documents' 
      })
    } finally {
      setActionLoading(null)
    }
  }

  async function handleDelete(documentId: string, documentName: string) {
    if (!confirm(`Delete "${documentName}"? This cannot be undone.`)) return
    
    setActionLoading(documentId)
    setActionMessage(null)
    try {
      const res = await authFetch(`/api/documents/${documentId}`, { method: 'DELETE' })
      if (!res.ok) {
        throw new Error(`Failed to delete: HTTP ${res.status}`)
      }
      setActionMessage({ 
        type: 'success', 
        text: `Deleted "${documentName}"` 
      })
      fetchDocuments()
    } catch (err) {
      setActionMessage({ 
        type: 'error', 
        text: err instanceof Error ? err.message : 'Failed to delete document' 
      })
    } finally {
      setActionLoading(null)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-8">
      <div className="w-full max-w-5xl">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-hiplink-dark dark:text-dark-text">Documents</h1>
            {rawRole && (
              <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim mt-1">
                {rawRole === 'admin' && 'Full document management access'}
                {rawRole === 'user' && 'Can view and upload documents'}
                {rawRole === 'viewer' && 'Read-only access'}
              </p>
            )}
          </div>
          <div className="flex items-center gap-2">
            {perms.canReindexDocuments && (
              <button
                onClick={handleReindexAll}
                disabled={actionLoading === 'reindex' || loading}
                className="px-4 py-2 rounded-lg font-medium bg-amber-500 text-white hover:bg-amber-600 transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                {actionLoading === 'reindex' ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Re-indexing...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Re-index All
                  </>
                )}
              </button>
            )}
            {perms.canUploadDocuments && (
              <Link
                href="/documents/upload"
                className="bg-hiplink-blue dark:bg-sky-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-hiplink-blue-dark dark:hover:bg-sky-500 transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                Upload Document
              </Link>
            )}
          </div>
        </div>

        {/* Action Message */}
        {actionMessage && (
          <div className={`mb-4 px-4 py-3 rounded-lg border ${
            actionMessage.type === 'success' 
              ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-700 dark:text-green-400'
              : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-700 dark:text-red-400'
          }`}>
            {actionMessage.text}
          </div>
        )}

        {/* Role Notice */}
        {rawRole === 'viewer' && (
          <div className="mb-4 px-4 py-3 rounded-lg bg-blue-50 dark:bg-sky-900/20 border border-blue-200 dark:border-sky-800">
            <p className="text-sm text-blue-700 dark:text-sky-400">
              <strong>Viewer mode:</strong> Read-only access to documents. You cannot upload, delete, or re-index documents.
            </p>
          </div>
        )}
        {rawRole === 'user' && (
          <div className="mb-4 px-4 py-3 rounded-lg bg-blue-50 dark:bg-sky-900/20 border border-blue-200 dark:border-sky-800">
            <p className="text-sm text-blue-700 dark:text-sky-400">
              <strong>Note:</strong> Standard users can upload documents but cannot delete or re-index documents.
            </p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="bg-white dark:bg-dark-card shadow-lg rounded-lg p-8 text-center">
            <div className="flex flex-col items-center space-y-3">
              <svg className="animate-spin h-8 w-8 text-hiplink-blue dark:text-sky-400" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span className="text-hiplink-secondary dark:text-dark-text-dim">Loading documents...</span>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="bg-white dark:bg-dark-card shadow-lg rounded-lg p-8">
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 px-4 py-3 rounded-lg">
              <p className="font-medium">Error</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
            <button
              onClick={fetchDocuments}
              className="mt-4 text-hiplink-blue dark:text-sky-400 hover:text-hiplink-blue-dark dark:hover:text-sky-300 font-medium"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Documents Table */}
        {!loading && !error && (
          <div className="bg-white dark:bg-dark-card shadow-lg rounded-lg overflow-hidden">
            {documents.length === 0 ? (
              <div className="p-8 text-center">
                <svg
                  className="w-16 h-16 text-gray-300 dark:text-dark-text-dim mx-auto mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <h3 className="text-lg font-medium text-hiplink-dark dark:text-dark-text mb-1">
                  No documents yet
                </h3>
                <p className="text-hiplink-secondary dark:text-dark-text-dim mb-4">
                  {perms.canUploadDocuments 
                    ? 'Upload your first document to get started.'
                    : 'No documents have been uploaded yet.'}
                </p>
                {perms.canUploadDocuments && (
                  <Link
                    href="/documents/upload"
                    className="inline-block bg-hiplink-blue dark:bg-sky-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-hiplink-blue-dark dark:hover:bg-sky-500 transition-colors"
                  >
                    Upload Document
                  </Link>
                )}
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-dark-border">
                  <thead className="bg-gray-50 dark:bg-dark-elevated">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">
                        Name
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">
                        MIME Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">
                        Size
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">
                        Created
                      </th>
                      {perms.canDeleteDocuments && (
                        <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">
                          Actions
                        </th>
                      )}
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-dark-card divide-y divide-gray-200 dark:divide-dark-border">
                    {documents.map((doc) => (
                      <tr key={doc.id} className="hover:bg-gray-50 dark:hover:bg-dark-elevated transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-hiplink-dark dark:text-dark-text">
                            {doc.original_name}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-hiplink-secondary dark:text-dark-text-dim">
                            {doc.mime_type}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-hiplink-secondary dark:text-dark-text-dim">
                            {formatBytes(doc.size_bytes)}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              doc.status === 'indexed'
                                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                                : doc.status === 'processing'
                                ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'
                                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                            }`}
                          >
                            {doc.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-hiplink-secondary dark:text-dark-text-dim">
                            {new Date(doc.created_at).toLocaleString()}
                          </div>
                        </td>
                        {perms.canDeleteDocuments && (
                          <td className="px-6 py-4 whitespace-nowrap text-right">
                            <button
                              onClick={() => handleDelete(doc.id, doc.original_name)}
                              disabled={actionLoading === doc.id}
                              className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300 font-medium text-sm disabled:opacity-50 flex items-center gap-1 ml-auto"
                            >
                              {actionLoading === doc.id ? (
                                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                              ) : (
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                              )}
                              Delete
                            </button>
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        <Link
          href="/chat"
          className="block mt-6 text-center text-hiplink-blue dark:text-sky-400 hover:text-hiplink-blue-dark dark:hover:text-sky-300 font-medium"
        >
          ← Back to Chat
        </Link>
      </div>
    </main>
  )
}