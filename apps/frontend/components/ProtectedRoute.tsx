'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'

interface ProtectedRouteProps {
  children: React.ReactNode
  /** Optional required permission — if set, user must have this permission */
  requirePermission?: 'canAccessObservability' | 'canAccessEvaluations' | 'canUseDebug' | 'canViewDocuments'
}

/**
 * Wraps protected route content.
 *
 * Behavior:
 * - Shows a loading spinner while auth config or token validation is in flight.
 * - If no valid JWT and not in dev mode, redirects to /auth.
 * - In dev mode, allows dev-user fallback.
 * - If requirePermission is set, redirects to / when user lacks the permission.
 */
export default function ProtectedRoute({ children, requirePermission }: ProtectedRouteProps) {
  const { auth, loading, configLoaded, authMode, devUser, user } = useAuth()
  const router = useRouter()

  const isAuthenticated =
    auth?.authenticated === true ||
    (authMode === 'dev' && Boolean(devUser))

  useEffect(() => {
    if (!configLoaded) return
    if (loading) return
    if (!isAuthenticated) {
      router.replace('/auth')
      return
    }
    // Permission gate
    if (requirePermission && user && auth?.permissions) {
      const perms = auth.permissions as unknown as Record<string, boolean>
      if (!perms[requirePermission]) {
        router.replace('/')
      }
    }
  }, [configLoaded, loading, isAuthenticated, requirePermission, user, auth, router])

  if (!configLoaded || loading) {
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

  if (!isAuthenticated) {
    return null
  }

  return <>{children}</>
}
