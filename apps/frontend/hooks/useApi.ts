/**
 * Auth-aware API fetch hook.
 * Automatically includes dev auth headers.
 */

import { useAuth } from '@/contexts/AuthContext'
import { getApiBaseUrl } from '@/lib/api'

export function useAuthFetch() {
  const { devUser } = useAuth()

  return async function authFetch(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<Response> {
    const headers: Record<string, string> = {
      ...(options.headers as Record<string, string> || {}),
    }

    if (devUser) {
      headers['X-Dev-User'] = devUser
    }

    return fetch(`${getApiBaseUrl()}${endpoint}`, {
      ...options,
      headers,
    })
  }
}