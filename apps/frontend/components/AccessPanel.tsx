'use client'

/**
 * Phase 13 Access Panel — admin-only modal for managing document visibility,
 * ownership, and shares. Reuses the Modal/Field/ModalActions pattern from
 * apps/frontend/app/admin/users/page.tsx.
 *
 * Backed by the new backend endpoints:
 *  - GET   /api/admin/documents/{id}/access
 *  - PATCH /api/admin/documents/{id}/access
 * and the existing user-picker endpoint:
 *  - GET   /api/admin/users
 */

import { useEffect, useState } from 'react'
import { getApiBaseUrl } from '@/lib/api'

interface AccessSummary {
  document_id: number
  visibility: 'private' | 'shared' | 'global'
  owner_user_id: number | null
  owner_username: string | null
  shared_user_ids: number[]
  shared_role_access: { role: string; access_level: 'view' | 'manage' }[]
}

interface UserRow {
  id: number
  username: string
  email: string
  full_name: string | null
  role: 'admin' | 'user' | 'viewer'
  is_active: boolean
}

interface AccessPanelProps {
  documentId: number
  documentName: string
  token: string
  onClose: () => void
  onSaved: () => void
}

const ALL_ROLES = ['admin', 'user', 'viewer'] as const

export default function AccessPanel({
  documentId,
  documentName,
  token,
  onClose,
  onSaved,
}: AccessPanelProps) {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [summary, setSummary] = useState<AccessSummary | null>(null)
  const [users, setUsers] = useState<UserRow[]>([])
  const [visibility, setVisibility] = useState<'private' | 'shared' | 'global'>('private')
  const [selectedUserIds, setSelectedUserIds] = useState<Set<number>>(new Set())
  const [selectedRoles, setSelectedRoles] = useState<Set<string>>(new Set())

  // Load access summary and user list on mount.
  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      setError(null)
      try {
        const [summaryRes, usersRes] = await Promise.all([
          fetch(`${getApiBaseUrl()}/api/admin/documents/${documentId}/access`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
          fetch(`${getApiBaseUrl()}/api/admin/users`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ])
        if (!summaryRes.ok) {
          throw new Error(`Failed to load access summary: ${summaryRes.status}`)
        }
        if (!usersRes.ok) {
          throw new Error(`Failed to load users: ${usersRes.status}`)
        }
        const summaryData: AccessSummary = await summaryRes.json()
        const usersData: UserRow[] = await usersRes.json()
        if (cancelled) return
        setSummary(summaryData)
        setUsers(usersData.filter((u) => u.is_active))
        setVisibility(summaryData.visibility)
        setSelectedUserIds(new Set(summaryData.shared_user_ids))
        setSelectedRoles(new Set(summaryData.shared_role_access.map((r) => r.role)))
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load access data')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [documentId, token])

  function toggleUser(userId: number) {
    setSelectedUserIds((prev) => {
      const next = new Set(prev)
      if (next.has(userId)) next.delete(userId)
      else next.add(userId)
      return next
    })
  }

  function toggleRole(role: string) {
    setSelectedRoles((prev) => {
      const next = new Set(prev)
      if (next.has(role)) next.delete(role)
      else next.add(role)
      return next
    })
  }

  async function handleSave() {
    setSaving(true)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/admin/documents/${documentId}/access`, {
        method: 'PATCH',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          visibility,
          shared_user_ids: Array.from(selectedUserIds),
          shared_roles: Array.from(selectedRoles),
          access_level: 'view',
        }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Save failed (${res.status})`)
      }
      onSaved()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save access changes')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div
        className="bg-white dark:bg-dark-card rounded-xl shadow-xl max-w-lg w-full p-6"
        data-testid="access-panel-modal"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">
            Manage Access
          </h2>
          <button
            onClick={onClose}
            className="text-hiplink-secondary dark:text-dark-text-muted hover:text-hiplink-dark dark:hover:text-dark-text"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted mb-4">
          <span className="font-medium text-hiplink-dark dark:text-dark-text">{documentName}</span>
          {summary?.owner_username && (
            <span className="ml-2">· owned by <span className="font-mono">{summary.owner_username}</span></span>
          )}
        </p>

        {error && (
          <div className="mb-4 px-4 py-3 rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-8 text-hiplink-secondary dark:text-dark-text-muted">
            <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Loading access data...
          </div>
        ) : (
          <div className="space-y-4 max-h-[60vh] overflow-y-auto">
            {/* Visibility dropdown */}
            <Field label="Visibility" required hint="Global: all authenticated users. Shared: explicit shares only. Private: owner + admin only.">
              <select
                value={visibility}
                onChange={(e) => setVisibility(e.target.value as 'private' | 'shared' | 'global')}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
                data-testid="access-panel-visibility"
              >
                <option value="private">Private</option>
                <option value="shared">Shared</option>
                <option value="global">Global</option>
              </select>
            </Field>

            {/* Share with users (multi-select) */}
            <Field label="Share with users" hint="Users listed here will get view access to this document.">
              <div className="border border-hiplink-border dark:border-dark-border rounded-lg max-h-40 overflow-y-auto bg-white dark:bg-dark-card">
                {users.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-hiplink-secondary dark:text-dark-text-dim">
                    No active users found.
                  </div>
                ) : (
                  users.map((u) => (
                    <label
                      key={u.id}
                      className="flex items-center gap-2 px-3 py-2 hover:bg-blue-50 dark:hover:bg-dark-elevated cursor-pointer text-sm"
                    >
                      <input
                        type="checkbox"
                        checked={selectedUserIds.has(u.id)}
                        onChange={() => toggleUser(u.id)}
                        className="rounded border-hiplink-border"
                      />
                      <span className="flex-1">
                        <span className="font-mono text-xs text-hiplink-dark dark:text-dark-text">{u.username}</span>
                        {u.full_name && (
                          <span className="text-hiplink-secondary dark:text-dark-text-muted"> · {u.full_name}</span>
                        )}
                      </span>
                      <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
                        {u.role}
                      </span>
                    </label>
                  ))
                )}
              </div>
            </Field>

            {/* Share with roles (multi-select) */}
            <Field label="Share with roles" hint="Every user with the selected role will get view access.">
              <div className="flex flex-wrap gap-2">
                {ALL_ROLES.map((r) => {
                  const checked = selectedRoles.has(r)
                  return (
                    <label
                      key={r}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer text-sm transition-colors ${
                        checked
                          ? 'bg-hiplink-blue text-white border-hiplink-blue'
                          : 'bg-white dark:bg-dark-card border-hiplink-border dark:border-dark-border text-hiplink-dark dark:text-dark-text hover:bg-blue-50 dark:hover:bg-dark-elevated'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleRole(r)}
                        className="sr-only"
                      />
                      {r}
                    </label>
                  )
                })}
              </div>
            </Field>
          </div>
        )}

        <div className="flex gap-2 pt-4 mt-4 border-t border-hiplink-border dark:border-dark-border">
          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            className="flex-1 px-4 py-2 rounded-lg font-medium bg-gray-100 dark:bg-dark-elevated text-hiplink-secondary dark:text-dark-text-muted hover:bg-gray-200 dark:hover:bg-dark-border transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={loading || saving}
            data-testid="access-panel-save"
            className="flex-1 px-4 py-2 rounded-lg font-medium bg-hiplink-blue text-white hover:bg-hiplink-blue-dark transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

function Field({
  label,
  required,
  hint,
  children,
}: {
  label: string
  required?: boolean
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-hiplink-dark dark:text-dark-text mb-1">
        {label}
        {required && <span className="text-red-500 ml-0.5">*</span>}
      </label>
      {children}
      {hint && <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mt-1">{hint}</p>}
    </div>
  )
}
