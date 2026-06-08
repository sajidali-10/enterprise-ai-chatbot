'use client'

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'

interface AuthInfo {
  authenticated: boolean
  username: string
  role: string
  user_id: number | null
  is_admin: boolean
  dev_mode: boolean
}

interface AuthContextType {
  auth: AuthInfo | null
  loading: boolean
  refreshAuth: () => Promise<void>
  devUser: string | null
  setDevUser: (userId: string | null) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<AuthInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [devUser, setDevUserState] = useState<string | null>(null)

  const refreshAuth = useCallback(async () => {
    try {
      // Dynamically import to avoid circular deps
      const { getApiBaseUrl } = await import('@/lib/api')
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (devUser) {
        headers['X-Dev-User'] = devUser
      }
      const res = await fetch(`${getApiBaseUrl()}/api/chat/auth-info`, { headers })
      if (res.ok) {
        const data = await res.json()
        setAuth(data)
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
    // Check localStorage on mount
    const stored = localStorage.getItem('dev_user')
    if (stored && !devUser) {
      setDevUserState(stored)
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

  return (
    <AuthContext.Provider value={{ auth, loading, refreshAuth, devUser, setDevUser }}>
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