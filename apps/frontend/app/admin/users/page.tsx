'use client'

import { useEffect, useState, useCallback } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { getApiBaseUrl } from '@/lib/api'
import ProtectedRoute from '@/components/ProtectedRoute'

interface UserRow {
  id: number
  username: string
  email: string
  full_name: string | null
  role: 'admin' | 'user' | 'viewer'
  is_active: boolean
  created_at: string | null
  updated_at: string | null
  last_login: string | null
}

interface FormState {
  username: string
  email: string
  password: string
  full_name: string
  role: 'admin' | 'user' | 'viewer'
  is_active: boolean
}

const EMPTY_FORM: FormState = {
  username: '',
  email: '',
  password: '',
  full_name: '',
  role: 'user',
  is_active: true,
}

function UsersPageInner() {
  const { auth } = useAuth()
  const [users, setUsers] = useState<UserRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editTarget, setEditTarget] = useState<UserRow | null>(null)
  const [resetTarget, setResetTarget] = useState<UserRow | null>(null)
  const [createForm, setCreateForm] = useState<FormState>(EMPTY_FORM)
  const [editForm, setEditForm] = useState<FormState>(EMPTY_FORM)
  const [newPassword, setNewPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [actionLoadingId, setActionLoadingId] = useState<number | null>(null)

  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null
  const authHeaders: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) {
    authHeaders['Authorization'] = `Bearer ${token}`
  }

  const fetchUsers = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/admin/users`, { headers: authHeaders })
      if (!res.ok) {
        throw new Error(`Failed to load users: HTTP ${res.status}`)
      }
      const data = await res.json()
      setUsers(data || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load users')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!auth?.authenticated) return
    fetchUsers()
  }, [auth?.authenticated, fetchUsers])

  function flashSuccess(msg: string) {
    setSuccess(msg)
    window.setTimeout(() => setSuccess(null), 3500)
  }

  function flashError(msg: string) {
    setError(msg)
    window.setTimeout(() => setError(null), 5000)
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/admin/users`, {
        method: 'POST',
        headers: authHeaders,
        body: JSON.stringify({
          username: createForm.username.trim(),
          email: createForm.email.trim(),
          password: createForm.password,
          full_name: createForm.full_name.trim() || null,
          role: createForm.role,
          is_active: createForm.is_active,
        }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      setShowCreateModal(false)
      setCreateForm(EMPTY_FORM)
      flashSuccess('User created successfully')
      fetchUsers()
    } catch (err) {
      flashError(err instanceof Error ? err.message : 'Failed to create user')
    } finally {
      setSubmitting(false)
    }
  }

  function openEdit(user: UserRow) {
    setEditTarget(user)
    setEditForm({
      username: user.username,
      email: user.email,
      password: '',
      full_name: user.full_name || '',
      role: user.role,
      is_active: user.is_active,
    })
  }

  async function handleUpdate(e: React.FormEvent) {
    e.preventDefault()
    if (!editTarget) return
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/admin/users/${editTarget.id}`, {
        method: 'PATCH',
        headers: authHeaders,
        body: JSON.stringify({
          email: editForm.email.trim(),
          full_name: editForm.full_name.trim() || null,
          role: editForm.role,
          is_active: editForm.is_active,
        }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      setEditTarget(null)
      flashSuccess('User updated successfully')
      fetchUsers()
    } catch (err) {
      flashError(err instanceof Error ? err.message : 'Failed to update user')
    } finally {
      setSubmitting(false)
    }
  }

  async function handleDeactivate(user: UserRow) {
    if (!confirm(`Deactivate "${user.username}"? They will not be able to log in.`)) return
    setActionLoadingId(user.id)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/admin/users/${user.id}`, {
        method: 'DELETE',
        headers: authHeaders,
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      flashSuccess(`User "${user.username}" deactivated`)
      fetchUsers()
    } catch (err) {
      flashError(err instanceof Error ? err.message : 'Failed to deactivate user')
    } finally {
      setActionLoadingId(null)
    }
  }

  async function handleReactivate(user: UserRow) {
    setActionLoadingId(user.id)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/admin/users/${user.id}`, {
        method: 'PATCH',
        headers: authHeaders,
        body: JSON.stringify({ is_active: true }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      flashSuccess(`User "${user.username}" reactivated`)
      fetchUsers()
    } catch (err) {
      flashError(err instanceof Error ? err.message : 'Failed to reactivate user')
    } finally {
      setActionLoadingId(null)
    }
  }

  async function handleResetPassword(e: React.FormEvent) {
    e.preventDefault()
    if (!resetTarget) return
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/admin/users/${resetTarget.id}/reset-password`, {
        method: 'POST',
        headers: authHeaders,
        body: JSON.stringify({ new_password: newPassword }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      setResetTarget(null)
      setNewPassword('')
      flashSuccess(`Password reset for "${resetTarget.username}"`)
    } catch (err) {
      flashError(err instanceof Error ? err.message : 'Failed to reset password')
    } finally {
      setSubmitting(false)
    }
  }

  const roleColors: Record<string, string> = {
    admin: 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400',
    user: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400',
    viewer: 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
  }

  return (
    <div className="min-h-screen bg-hiplink-background dark:bg-dark-bg">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-hiplink-dark dark:text-dark-text">User Management</h1>
            <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim mt-1">
              Manage local users, roles, and passwords.
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary px-4 py-2 rounded-lg font-medium flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Create User
          </button>
        </div>

        {error && (
          <div className="mb-4 px-4 py-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm">
            {error}
          </div>
        )}
        {success && (
          <div className="mb-4 px-4 py-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-400 text-sm">
            {success}
          </div>
        )}

        <div className="bg-white dark:bg-dark-card shadow-lg rounded-lg overflow-hidden">
          {loading ? (
            <div className="p-8 text-center text-hiplink-secondary dark:text-dark-text-dim">Loading users…</div>
          ) : users.length === 0 ? (
            <div className="p-8 text-center text-hiplink-secondary dark:text-dark-text-dim">No users yet.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-dark-border">
                <thead className="bg-gray-50 dark:bg-dark-elevated">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">Username</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">Email</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">Full Name</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">Role</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">Last Login</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-dark-text-dim uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-dark-card divide-y divide-gray-200 dark:divide-dark-border">
                  {users.map((user) => (
                    <tr key={user.id} className="hover:bg-gray-50 dark:hover:bg-dark-elevated transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-hiplink-dark dark:text-dark-text">{user.username}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-hiplink-secondary dark:text-dark-text-dim">{user.email}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-hiplink-secondary dark:text-dark-text-dim">{user.full_name || '—'}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${roleColors[user.role]}`}>{user.role}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {user.is_active ? (
                          <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">Active</span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400">Inactive</span>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-hiplink-secondary dark:text-dark-text-dim">
                        {user.last_login ? new Date(user.last_login).toLocaleString() : 'Never'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm space-x-2">
                        <button
                          onClick={() => openEdit(user)}
                          className="text-hiplink-blue dark:text-sky-400 hover:text-hiplink-blue-dark dark:hover:text-sky-300 font-medium"
                        >
                          Edit
                        </button>
                        <button
                          onClick={() => setResetTarget(user)}
                          className="text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300 font-medium"
                        >
                          Reset PW
                        </button>
                        {user.is_active ? (
                          <button
                            onClick={() => handleDeactivate(user)}
                            disabled={actionLoadingId === user.id}
                            className="text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 font-medium disabled:opacity-50"
                          >
                            Deactivate
                          </button>
                        ) : (
                          <button
                            onClick={() => handleReactivate(user)}
                            disabled={actionLoadingId === user.id}
                            className="text-green-600 dark:text-green-400 hover:text-green-700 dark:hover:text-green-300 font-medium disabled:opacity-50"
                          >
                            Reactivate
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Create User Modal */}
      {showCreateModal && (
        <Modal title="Create User" onClose={() => setShowCreateModal(false)}>
          <form onSubmit={handleCreate} className="space-y-4">
            <Field label="Username" required>
              <input
                type="text"
                required
                minLength={1}
                maxLength={100}
                value={createForm.username}
                onChange={(e) => setCreateForm({ ...createForm, username: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
              />
            </Field>
            <Field label="Email" required>
              <input
                type="email"
                required
                value={createForm.email}
                onChange={(e) => setCreateForm({ ...createForm, email: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
              />
            </Field>
            <Field label="Password" required hint="Minimum 8 characters">
              <input
                type="password"
                required
                minLength={8}
                value={createForm.password}
                onChange={(e) => setCreateForm({ ...createForm, password: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
              />
            </Field>
            <Field label="Full Name">
              <input
                type="text"
                value={createForm.full_name}
                onChange={(e) => setCreateForm({ ...createForm, full_name: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
              />
            </Field>
            <Field label="Role" required>
              <select
                value={createForm.role}
                onChange={(e) => setCreateForm({ ...createForm, role: e.target.value as FormState['role'] })}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
              >
                <option value="user">User</option>
                <option value="viewer">Viewer</option>
                <option value="admin">Admin</option>
              </select>
            </Field>
            <label className="flex items-center gap-2 text-sm text-hiplink-dark dark:text-dark-text">
              <input
                type="checkbox"
                checked={createForm.is_active}
                onChange={(e) => setCreateForm({ ...createForm, is_active: e.target.checked })}
              />
              Active
            </label>
            <ModalActions
              submitting={submitting}
              onCancel={() => setShowCreateModal(false)}
              submitLabel="Create User"
            />
          </form>
        </Modal>
      )}

      {/* Edit User Modal */}
      {editTarget && (
        <Modal title={`Edit User: ${editTarget.username}`} onClose={() => setEditTarget(null)}>
          <form onSubmit={handleUpdate} className="space-y-4">
            <Field label="Email" required>
              <input
                type="email"
                required
                value={editForm.email}
                onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
              />
            </Field>
            <Field label="Full Name">
              <input
                type="text"
                value={editForm.full_name}
                onChange={(e) => setEditForm({ ...editForm, full_name: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
              />
            </Field>
            <Field label="Role" required>
              <select
                value={editForm.role}
                onChange={(e) => setEditForm({ ...editForm, role: e.target.value as FormState['role'] })}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
              >
                <option value="user">User</option>
                <option value="viewer">Viewer</option>
                <option value="admin">Admin</option>
              </select>
            </Field>
            <label className="flex items-center gap-2 text-sm text-hiplink-dark dark:text-dark-text">
              <input
                type="checkbox"
                checked={editForm.is_active}
                onChange={(e) => setEditForm({ ...editForm, is_active: e.target.checked })}
              />
              Active
            </label>
            <ModalActions
              submitting={submitting}
              onCancel={() => setEditTarget(null)}
              submitLabel="Save Changes"
            />
          </form>
        </Modal>
      )}

      {/* Reset Password Modal */}
      {resetTarget && (
        <Modal title={`Reset Password: ${resetTarget.username}`} onClose={() => setResetTarget(null)}>
          <form onSubmit={handleResetPassword} className="space-y-4">
            <Field label="New Password" required hint="Minimum 8 characters">
              <input
                type="password"
                required
                minLength={8}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-hiplink-border dark:border-dark-border bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
                autoFocus
              />
            </Field>
            <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim">
              The new password is not displayed after submission. Communicate it to the user through a secure channel.
            </p>
            <ModalActions
              submitting={submitting}
              onCancel={() => { setResetTarget(null); setNewPassword('') }}
              submitLabel="Reset Password"
            />
          </form>
        </Modal>
      )}
    </div>
  )
}

function Modal({ title, children, onClose }: { title: string; children: React.ReactNode; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className="bg-white dark:bg-dark-card rounded-xl shadow-xl max-w-md w-full p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">{title}</h2>
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
        {children}
      </div>
    </div>
  )
}

function Field({ label, required, hint, children }: { label: string; required?: boolean; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium text-hiplink-dark dark:text-dark-text mb-1">
        {label}{required && <span className="text-red-500 ml-0.5">*</span>}
      </label>
      {children}
      {hint && <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mt-1">{hint}</p>}
    </div>
  )
}

function ModalActions({ submitting, onCancel, submitLabel }: { submitting: boolean; onCancel: () => void; submitLabel: string }) {
  return (
    <div className="flex gap-2 pt-2">
      <button
        type="submit"
        disabled={submitting}
        className="btn-primary px-4 py-2 rounded-lg font-medium disabled:opacity-50"
      >
        {submitting ? 'Working…' : submitLabel}
      </button>
      <button
        type="button"
        onClick={onCancel}
        className="btn-secondary px-4 py-2 rounded-lg font-medium"
      >
        Cancel
      </button>
    </div>
  )
}

export default function UsersPage() {
  return (
    <ProtectedRoute requirePermission="canManageUsers">
      <UsersPageInner />
    </ProtectedRoute>
  )
}
