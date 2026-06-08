/**
 * Get the API base URL dynamically based on browser environment.
 * This avoids hardcoding localhost or IP addresses.
 * 
 * If NEXT_PUBLIC_API_BASE_URL is set, use it.
 * Otherwise, construct URL from window.location.protocol and window.location.hostname
 * using the same hostname but port 8000 (backend port).
 */
export function getApiBaseUrl(): string {
  // If explicitly configured, use that value
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL;
  }

  // In browser context, dynamically determine the backend URL
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    // Use same host but port 8000 for backend
    return `${protocol}//${hostname}:8000`;
  }

  // Fallback for SSR (should not be used for browser requests)
  return 'http://localhost:8000';
}