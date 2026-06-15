'use client'

/**
 * Backwards-compatible re-export of the single shared authenticated header.
 * New code should import `AppHeader` from `@/components/AppHeader` directly.
 */
import AppHeader from './AppHeader'

interface HeaderProps {
  /** @deprecated ignored — visibility is driven by current user permissions */
  showAdminNav?: boolean
}

export default function Header(_props: HeaderProps = {}) {
  return <AppHeader />
}
