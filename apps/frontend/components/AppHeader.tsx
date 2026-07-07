'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { useTheme } from '@/contexts/ThemeContext'
import { normalizePermissions, type PermissionFlags } from '@/lib/permissions'
import HipLinkLogo from './HipLinkLogo'

interface NavLinkProps {
  href: string
  children: React.ReactNode
  isActive: boolean
}

function NavLink({ href, children, isActive }: NavLinkProps) {
  return (
    <Link
      href={href}
      className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
        isActive
          ? 'bg-hiplink-blue text-white'
          : 'text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 dark:hover:bg-dark-elevated dark:text-dark-text-muted'
      }`}
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

interface NavItem {
  href: string
  label: string
  /** Permission required to show this item; omit to always show to authenticated users */
  permission?: keyof PermissionFlags
  /** Match function against pathname; defaults to exact match on href */
  matchPath?: (pathname: string) => boolean
}

/**
 * Single source of truth for the authenticated top navigation.
 * Order MUST stay:
 *   1. Dashboard
 *   2. Chat
 *   3. Documents
 *   4. Observability
 *   5. Evaluations
 *   6. Users
 *   7. Security
 *   8. Audit Logs
 */
const NAV_ITEMS: NavItem[] = [
  { href: '/', label: 'Dashboard', matchPath: (p) => p === '/' },
  { href: '/chat', label: 'Chat', matchPath: (p) => p === '/chat' },
  {
    href: '/documents',
    label: 'Documents',
    permission: 'canViewDocuments',
    matchPath: (p) => p === '/documents' || p.startsWith('/documents/'),
  },
  {
    href: '/admin/observability',
    label: 'Observability',
    permission: 'canAccessObservability',
    matchPath: (p) => p === '/admin/observability',
  },
  {
    href: '/admin/evaluations',
    label: 'Evaluations',
    permission: 'canAccessEvaluations',
    matchPath: (p) => p === '/admin/evaluations',
  },
  {
    href: '/admin/users',
    label: 'Users',
    permission: 'canManageUsers',
    matchPath: (p) => p === '/admin/users' || p.startsWith('/admin/users/'),
  },
  {
    href: '/admin/security',
    label: 'Security',
    permission: 'canAccessObservability',
    matchPath: (p) => p === '/admin/security',
  },
  {
    href: '/admin/audit-logs',
    label: 'Audit Logs',
    permission: 'canAccessObservability',
    matchPath: (p) => p === '/admin/audit-logs',
  },
  {
    href: '/admin/rag',
    label: 'RAG Config',
    permission: 'canAccessObservability',
    matchPath: (p) => p === '/admin/rag',
  },
]

export default function AppHeader() {
  const pathname = usePathname()
  const { auth, user, logout } = useAuth()

  const rawRole = (auth?.role ?? 'viewer') as string
  const perms: PermissionFlags = normalizePermissions(auth?.permissions, rawRole)
  const isAuthenticated = auth?.authenticated === true

  const displayName = user?.full_name || user?.username || user?.email || auth?.username || ''

  return (
    <header className="brand-header flex-shrink-0">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-3">
              <HipLinkLogo variant="auto" width={36} height={36} priority />
              <span className="text-lg font-semibold text-hiplink-dark dark:text-dark-text">
                HipLink AI Assistant
              </span>
            </Link>
          </div>

          {isAuthenticated && (
            <nav className="flex items-center gap-1" data-testid="primary-nav">
              {NAV_ITEMS.map((item) => {
                // Hide items the current user is not entitled to.
                if (item.permission && !perms[item.permission]) return null
                const isActive = item.matchPath
                  ? item.matchPath(pathname)
                  : pathname === item.href
                return (
                  <NavLink key={item.href} href={item.href} isActive={isActive}>
                    {item.label}
                  </NavLink>
                )
              })}

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
