'use client'

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'
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
  refreshAuth: () => Promise<void>
  logout: () => void
  devUser: string | null
  setDevUser: (userId: string | null) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

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

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<AuthInfo | null>(null)
  const [user, setUser] = useState<UserInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [devUser, setDevUserState] = useState<string | null>(null)

  const refreshAuth = useCallback(async () => {
    try {
      const token = localStorage.getItem('access_token')
      const devUserLocal = localStorage.getItem('dev_user')

      if (token) {
        // Try JWT auth first
        const res = await fetch(`${getApiBaseUrl()}/api/auth/me`, {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
        })
        if (res.ok) {
          const data: UserInfo = await res.json()
          setUser(data)
          setAuth(buildAuthInfoFromUser(data))
          setLoading(false)
          return
        }
        // Token invalid/expired — clear it
        localStorage.removeItem('access_token')
      }

      // Fall back to dev mode auth
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (devUserLocal) {
        headers['X-Dev-User'] = devUserLocal
      }
      const res = await fetch(`${getApiBaseUrl()}/api/chat/auth-info`, { headers })
      if (res.ok) {
        const data = await res.json()
        setAuth(data)
        if (devUserLocal && !devUser) {
          setDevUserState(devUserLocal)
        }
      }
    } catch (err) {
      console.error('Failed to fetch auth info:', err)
    } finally {
      setLoading(false)
    }
  }, [devUser])

  useEffect(() => {
    refreshAuth()
  }, [refreshAuth])

  useEffect(() => {
    // Sync devUser state with localStorage on mount
    const storedDev = localStorage.getItem('dev_user')
    if (storedDev && !devUser) {
      setDevUserState(storedDev)
    }
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
    window.location.href = '/auth'
  }, [])

  return (
    <AuthContext.Provider value={{ auth, user, loading, refreshAuth, logout, devUser, setDevUser }}>
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
