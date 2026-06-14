/**
 * Auth-aware API fetch hook.
 * Automatically includes JWT Bearer token when present.
 *
 * Dev-mode X-Dev-User header is only included when the backend reports
 * AUTH_MODE=dev. In AUTH_MODE=local, only JWT auth is allowed.
 */

import { useAuth } from '@/contexts/AuthContext'
import { getApiBaseUrl } from '@/lib/api'

export function useAuthFetch() {
  const { authMode, devUser } = useAuth()

  return async function authFetch(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<Response> {
    const headers: Record<string, string> = {
      ...(options.headers as Record<string, string> || {}),
    }

    const token = localStorage.getItem('access_token')
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    } else if (authMode === 'dev' && devUser) {
      headers['X-Dev-User'] = devUser
    }

    return fetch(`${getApiBaseUrl()}${endpoint}`, {
      ...options,
      headers,
    })
  }
}
