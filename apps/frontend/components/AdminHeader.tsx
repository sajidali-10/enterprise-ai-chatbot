'use client'

import Link from 'next/link'
import Image from 'next/image'
import { usePathname } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { useTheme } from '@/contexts/ThemeContext'

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  
  return (
    <button
      onClick={toggleTheme}
      className="p-2 rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 dark:hover:bg-dark-elevated dark:text-dark-text-muted transition-colors"
      title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      {theme === 'light' ? (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
        </svg>
      ) : (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
      )}
    </button>
  )
}

export default function AdminHeader() {
  const pathname = usePathname()
  const { devUser } = useAuth()

  const isActive = (path: string) => {
    return pathname === path
  }

  return (
    <header className="brand-header flex-shrink-0">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-3">
              <Image
                src="/hiplink-logo.png"
                alt="HipLink"
                width={36}
                height={36}
                className="object-contain"
              />
              <div>
                <span className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">HipLink AI Assistant</span>
                <span className="text-hiplink-secondary dark:text-dark-text-dim mx-2">•</span>
                <span className="text-sm font-medium text-hiplink-secondary dark:text-dark-text-muted">Admin</span>
              </div>
            </Link>
          </div>
          <nav className="flex items-center gap-2">
            <Link
              href="/chat"
              className="px-4 py-2 text-sm font-medium rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 dark:hover:bg-dark-elevated dark:text-dark-text-muted transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              Chat
            </Link>
            <Link
              href="/admin/observability"
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                isActive('/admin/observability')
                  ? 'bg-hiplink-blue text-white'
                  : 'text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 dark:hover:bg-dark-elevated dark:text-dark-text-muted'
              }`}
            >
              Observability
            </Link>
            <Link
              href="/admin/evaluations"
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                isActive('/admin/evaluations')
                  ? 'bg-hiplink-blue text-white'
                  : 'text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 dark:hover:bg-dark-elevated dark:text-dark-text-muted'
              }`}
            >
              Evaluations
            </Link>
            <Link
              href="/documents"
              className="px-4 py-2 text-sm font-medium rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 dark:hover:bg-dark-elevated dark:text-dark-text-muted transition-colors"
            >
              Documents
            </Link>
            <div className="ml-2 flex items-center space-x-2">
              <ThemeToggle />
              <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-400`}>
                <span className="w-2 h-2 rounded-full bg-hiplink-success" />
                <span>{devUser || 'Guest'}</span>
              </div>
            </div>
          </nav>
        </div>
      </div>
    </header>
  )
}