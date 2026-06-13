/**
 * Authenticated API client for frontend requests.
 * Automatically includes dev auth headers when set.
 */

import { getApiBaseUrl } from './api'

export interface RequestOptions extends RequestInit {
  useAuth?: boolean
}

export async function apiRequest<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { useAuth = true, ...fetchOptions } = options
  
  const headers: Record<string, string> = {
    ...(fetchOptions.headers as Record<string, string> || {}),
  }
  
  // Add dev auth header if available and requested
  if (useAuth) {
    const devUser = localStorage.getItem('dev_user')
    if (devUser) {
      headers['X-Dev-User'] = devUser
    }
  }
  
  const url = `${getApiBaseUrl()}${endpoint}`
  
  const response = await fetch(url, {
    ...fetchOptions,
    headers,
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  
  return response.json()
}

// Convenience methods
export const api = {
  get: <T>(endpoint: string, options?: RequestOptions) =>
    apiRequest<T>(endpoint, { ...options, method: 'GET' }),
  
  post: <T>(endpoint: string, body?: unknown, options?: RequestOptions) =>
    apiRequest<T>(endpoint, {
      ...options,
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'Content-Type': 'application/json', ...options?.headers },
    }),
  
  postForm: <T>(endpoint: string, formData: FormData, options?: RequestOptions) =>
    apiRequest<T>(endpoint, {
      ...options,
      method: 'POST',
      body: formData,
      // Don't set Content-Type for FormData - browser sets it with boundary
      headers: options?.headers,
    }),
}