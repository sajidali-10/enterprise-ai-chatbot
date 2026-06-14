'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'

interface ProtectedRouteProps {
  children: React.ReactNode
  /** Optional required permission — if set, user must have this permission */
  requirePermission?: 'canAccessObservability' | 'canAccessEvaluations' | 'canUseDebug' | 'canViewDocuments' | 'canManageUsers'
}

/**
 * Wraps protected route content.
 *
 * Behavior:
 * - Shows a loading spinner while the auth check is in flight (loading=true).
 * - Once auth check completes: if not authenticated, redirects to /auth.
 * - If requirePermission is set, redirects to / when user lacks the permission.
 */
export default function ProtectedRoute({ children, requirePermission }: ProtectedRouteProps) {
  const { auth, loading, authMode, devUser, user } = useAuth()
  const router = useRouter()

  // Authenticated means: JWT auth succeeded, or (dev mode + dev user selected)
  const isAuthenticated =
    auth?.authenticated === true ||
    (authMode === 'dev' && Boolean(devUser))

  useEffect(() => {
    // Wait until loading is complete before deciding to redirect
    if (loading) return

    if (!isAuthenticated) {
      console.debug('[guard] redirecting to /auth (not authenticated)')
      router.replace('/auth')
      return
    }

    // Permission gate
    if (requirePermission && user && auth?.permissions) {
      const perms = auth.permissions as unknown as Record<string, boolean>
      if (!perms[requirePermission]) {
        console.debug('[guard] redirecting to / (missing permission)', requirePermission)
        router.replace('/')
      }
    }
  }, [loading, isAuthenticated, requirePermission, user, auth, router])

  // Show loading spinner while auth check is in flight
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-hiplink-background dark:bg-dark-bg">
        <div className="flex flex-col items-center gap-3">
          <svg className="animate-spin h-8 w-8 text-hiplink-blue dark:text-sky-400" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">Checking session…</p>
        </div>
      </div>
    )
  }

  // Loading is complete but not authenticated — render nothing while redirect happens
  if (!isAuthenticated) {
    return null
  }

  return <>{children}</>
}
