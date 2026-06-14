'use client'

import { createContext, useContext, useState, useEffect, useCallback, useRef, ReactNode } from 'react'
import { getApiBaseUrl } from '@/lib/api'
import { normalizePermissions, type PermissionFlags } from '@/lib/permissions'

export interface UserPermissions {
  can_use_general_chat: boolean
  can_use_knowledge_base: boolean
  can_use_debug: boolean
  can_view_documents: boolean
  can_upload_documents: boolean
  can_reindex_documents: boolean
  can_delete_documents: boolean
  can_access_observability: boolean
  can_access_evaluations: boolean
  can_submit_feedback: boolean
  can_manage_users: boolean
}

export interface UserInfo {
  id: number
  username: string
  email: string
  full_name: string | null
  role: string
  is_active: boolean
  permissions: UserPermissions
}

export type AuthMode = 'local' | 'dev'

interface AuthInfo {
  authenticated: boolean
  username: string
  role: string
  user_id: number | null
  is_admin: boolean
  dev_mode: boolean
  permissions: PermissionFlags
}

interface AuthContextType {
  auth: AuthInfo | null
  user: UserInfo | null
  loading: boolean
  /** Whether the auth config has been loaded from the backend */
  configLoaded: boolean
  /** Backend auth mode ('local' = production JWT login, 'dev' = dev header fallback) */
  authMode: AuthMode
  /** Whether dev-mode user switching is enabled (only true when AUTH_MODE=dev) */
  devAuthEnabled: boolean
  refreshAuth: () => Promise<void>
  loginWithToken: (token: string) => Promise<void>
  logout: () => void
  devUser: string | null
  setDevUser: (userId: string | null) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

/** Timeout for /api/auth/me — fail gracefully if backend is unreachable */
const ME_TIMEOUT_MS = 8000

function buildAuthInfoFromUser(user: UserInfo): AuthInfo {
  return {
    authenticated: true,
    username: user.username,
    role: user.role,
    user_id: user.id,
    is_admin: user.role === 'admin',
    dev_mode: false,
    permissions: normalizePermissions(user.permissions, user.role),
  }
}

/** Fetch with a timeout that rejects instead of hanging forever */
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = ME_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { ...options, signal: controller.signal })
    return res
  } finally {
    window.clearTimeout(timeoutId)
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<AuthInfo | null>(null)
  const [user, setUser] = useState<UserInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [configLoaded, setConfigLoaded] = useState(false)
  const [authMode, setAuthMode] = useState<AuthMode>('local')
  const [devAuthEnabled, setDevAuthEnabled] = useState(false)
  const [devUser, setDevUserState] = useState<string | null>(null)
  const initRanRef = useRef(false)

  // Fetch auth config from backend on mount
  useEffect(() => {
    let cancelled = false
    async function loadConfig() {
      try {
        console.debug('[auth] config: loading')
        const res = await fetchWithTimeout(`${getApiBaseUrl()}/api/auth/config`, {}, 5000)
        if (res.ok) {
          const data = await res.json()
          if (cancelled) return
          setAuthMode(data.auth_mode === 'dev' ? 'dev' : 'local')
          setDevAuthEnabled(Boolean(data.dev_auth_enabled))
          console.debug('[auth] config: loaded', { authMode: data.auth_mode })
        } else {
          console.debug('[auth] config: non-ok response', res.status)
        }
      } catch (err) {
        console.debug('[auth] config: failed, defaulting to local', err)
        // Default to local if config endpoint unreachable
      } finally {
        if (!cancelled) setConfigLoaded(true)
      }
    }
    loadConfig()
    return () => {
      cancelled = true
    }
  }, [])

  const refreshAuth = useCallback(async () => {
    try {
      console.debug('[auth] init: started')
      const token = localStorage.getItem('access_token')
      console.debug('[auth] init: token found', Boolean(token))

      if (token) {
        // Try JWT auth first
        const res = await fetchWithTimeout(`${getApiBaseUrl()}/api/auth/me`, {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
        })
        if (res.ok) {
          const data: UserInfo = await res.json()
          setUser(data)
          setAuth(buildAuthInfoFromUser(data))
          // Clear any stale dev_user when JWT auth succeeds
          localStorage.removeItem('dev_user')
          setDevUserState(null)
          console.debug('[auth] init: /me success', { username: data.username, role: data.role })
          return
        }
        console.debug('[auth] init: /me failed, clearing token', res.status)
        // Token invalid/expired — clear it
        localStorage.removeItem('access_token')
      }

      // No valid JWT — clear auth state
      setUser(null)
      setAuth(null)
    } catch (err) {
      console.debug('[auth] init: /me error, clearing token', err)
      // Backend unreachable or timeout — clear token to avoid infinite loading
      localStorage.removeItem('access_token')
      setUser(null)
      setAuth(null)
    } finally {
      console.debug('[auth] init: complete, loading=false')
      setLoading(false)
    }
  }, [])

  // Run initial auth validation on mount — this is critical for the session restore flow
  useEffect(() => {
    if (initRanRef.current) return
    initRanRef.current = true
    refreshAuth()
  }, [refreshAuth])

  const loginWithToken = useCallback(async (token: string) => {
    localStorage.setItem('access_token', token)
    localStorage.removeItem('dev_user')
    setDevUserState(null)
    // Fetch /me with the new token to populate auth state
    const res = await fetchWithTimeout(`${getApiBaseUrl()}/api/auth/me`, {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
    })
    if (!res.ok) {
      localStorage.removeItem('access_token')
      throw new Error('Failed to validate token')
    }
    const data: UserInfo = await res.json()
    setUser(data)
    setAuth(buildAuthInfoFromUser(data))
    setLoading(false)
  }, [])

  // Sync devUser from localStorage on mount (dev mode only)
  useEffect(() => {
    const storedDev = localStorage.getItem('dev_user')
    if (storedDev) setDevUserState(storedDev)
  }, [])

  const setDevUser = useCallback((userId: string | null) => {
    if (userId) {
      localStorage.setItem('dev_user', userId)
    } else {
      localStorage.removeItem('dev_user')
    }
    setDevUserState(userId)
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('dev_user')
    setUser(null)
    setAuth(null)
    setDevUserState(null)
    // Notify backend logout (best-effort)
    fetch(`${getApiBaseUrl()}/api/auth/logout`, { method: 'POST' }).catch(() => {})
    if (typeof window !== 'undefined') {
      window.location.href = '/auth'
    }
  }, [])

  return (
    <AuthContext.Provider value={{
      auth,
      user,
      loading,
      configLoaded,
      authMode,
      devAuthEnabled,
      refreshAuth,
      loginWithToken,
      logout,
      devUser,
      setDevUser,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
