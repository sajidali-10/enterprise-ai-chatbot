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
  
  // Add JWT Bearer token or dev auth header if available and requested
  if (useAuth) {
    const token = localStorage.getItem('access_token')
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    } else {
      const devUser = localStorage.getItem('dev_user')
      if (devUser) {
        headers['X-Dev-User'] = devUser
      }
    }
  }
  
  const url = `${getApiBaseUrl()}${endpoint}`
  
  const response = await fetch(url, {
    ...fetchOptions,
    headers,
  })
  
  if (!response.ok) {
    if (response.status === 401) {
      // Token expired or invalidated — clear and redirect to login
      localStorage.removeItem('access_token')
      if (typeof window !== 'undefined') {
        window.location.href = '/auth'
      }
      throw new Error('Session expired. Please log in again.')
    }
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