'use client'

import Link from 'next/link'
import Image from 'next/image'
import { usePathname } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { useTheme } from '@/contexts/ThemeContext'
import { normalizePermissions, type PermissionFlags } from '@/lib/permissions'

interface NavLinkProps {
  href: string
  children: React.ReactNode
  isActive?: boolean
  className?: string
}

function NavLink({ href, children, isActive, className = '' }: NavLinkProps) {
  return (
    <Link
      href={href}
      className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
        isActive
          ? 'bg-hiplink-blue text-white'
          : 'text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 dark:hover:bg-dark-elevated dark:text-dark-text-muted'
      } ${className}`}
    >
      {children}
    </Link>
  )
}

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

function RoleBadge({ role }: { role: string }) {
  const colors: Record<string, string> = {
    admin: 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400',
    user: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400',
    viewer: 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
  }

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[role] ?? colors.viewer}`}>
      {role}
    </span>
  )
}

interface HeaderProps {
  showAdminNav?: boolean
}

export default function Header({ showAdminNav = false }: HeaderProps) {
  const pathname = usePathname()
  const { auth, user, logout } = useAuth()

  // Determine effective role and permissions
  const rawRole = (auth?.role ?? 'viewer') as string
  const perms: PermissionFlags = normalizePermissions(auth?.permissions, rawRole)
  const isAuthenticated = auth?.authenticated === true

  const isActive = (path: string) => {
    if (path === '/chat') return pathname === '/chat'
    if (path === '/documents') return pathname === '/documents' || pathname.startsWith('/documents/')
    if (path === '/admin/observability') return pathname === '/admin/observability'
    if (path === '/admin/evaluations') return pathname === '/admin/evaluations'
    return pathname === path
  }

  const displayName = user?.full_name || user?.username || user?.email || auth?.username || ''

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
              <span className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">HipLink AI Assistant</span>
            </Link>
          </div>
          {isAuthenticated && (
            <nav className="flex items-center gap-1">
              {/* Chat - available to all authenticated roles */}
              <NavLink href="/chat" isActive={isActive('/chat')}>Chat</NavLink>

              {/* Documents - available if permission allows */}
              {perms.canViewDocuments && (
                <NavLink href="/documents" isActive={isActive('/documents')}>Documents</NavLink>
              )}

              {/* Admin navigation - only for admin users via explicit prop */}
              {showAdminNav && perms.canAccessObservability && (
                <>
                  <NavLink href="/admin/observability" isActive={isActive('/admin/observability')}>Observability</NavLink>
                  <NavLink href="/admin/evaluations" isActive={isActive('/admin/evaluations')}>Evaluations</NavLink>
                </>
              )}

              <div className="ml-2 flex items-center space-x-2">
                <ThemeToggle />
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-hiplink-dark dark:text-dark-text">
                    {displayName}
                  </span>
                  <RoleBadge role={rawRole} />
                  <button
                    onClick={logout}
                    className="px-3 py-1.5 rounded-lg text-sm bg-gray-100 dark:bg-dark-elevated text-hiplink-secondary dark:text-dark-text-muted hover:bg-gray-200 dark:hover:bg-dark-border transition-colors"
                  >
                    Logout
                  </button>
                </div>
              </div>
            </nav>
          )}
          {!isAuthenticated && (
            <div className="flex items-center gap-2">
              <ThemeToggle />
              <Link
                href="/auth"
                className="px-4 py-2 rounded-lg text-sm bg-hiplink-blue text-white hover:bg-hiplink-blue-dark transition-colors"
              >
                Sign In
              </Link>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
