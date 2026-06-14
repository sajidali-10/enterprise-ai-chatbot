'use client'

import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import { normalizePermissions } from '@/lib/permissions'
import AdminHeader from '@/components/AdminHeader'
import ProtectedRoute from '@/components/ProtectedRoute'

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { auth } = useAuth()
  const perms = normalizePermissions(auth?.permissions, auth?.role)
  const hasAdminAccess = (auth?.is_admin ?? false) || perms.canAccessObservability

  if (!hasAdminAccess) {
    return (
      <ProtectedRoute>
        <div className="min-h-screen bg-hiplink-background dark:bg-dark-bg flex items-center justify-center p-4">
          <div className="card dark:bg-dark-card p-8 text-center max-w-md">
            <div className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-hiplink-error dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h1 className="text-xl font-bold text-hiplink-dark dark:text-dark-text mb-2">Access Denied</h1>
            <p className="text-hiplink-secondary dark:text-dark-text-dim mb-4">You do not have permission to view this admin page.</p>
            <Link href="/" className="btn-primary inline-block">
              &larr; Back to Home
            </Link>
          </div>
        </div>
      </ProtectedRoute>
    )
  }

  return (
    <ProtectedRoute requirePermission="canAccessObservability">
      <div className="min-h-screen bg-hiplink-background dark:bg-dark-bg flex flex-col">
        <AdminHeader />
        <div className="flex-1 max-w-6xl mx-auto w-full px-4 py-6">
          {children}
        </div>
      </div>
    </ProtectedRoute>
  )
}
