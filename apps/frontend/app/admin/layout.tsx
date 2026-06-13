'use client'

import Link from 'next/link'
import Image from 'next/image'
import { useAuth } from '@/contexts/AuthContext'

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { auth } = useAuth()
  const isAdmin = auth?.is_admin ?? false

  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-hiplink-background flex items-center justify-center p-4">
        <div className="card p-8 text-center max-w-md">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-hiplink-error" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h1 className="text-xl font-bold text-hiplink-dark mb-2">Access Denied</h1>
          <p className="text-hiplink-secondary mb-4">Admin access required.</p>
          <Link href="/chat" className="btn-primary inline-block">
            &larr; Back to Chat
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-hiplink-background flex flex-col">
      {/* Admin Header */}
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
                  <span className="text-lg font-semibold text-hiplink-dark">HipLink AI Assistant</span>
                  <span className="text-hiplink-secondary mx-2">•</span>
                  <span className="text-sm font-medium text-hiplink-secondary">Admin</span>
                </div>
              </Link>
            </div>
            <nav className="flex items-center gap-2">
              <Link
                href="/chat"
                className="px-4 py-2 text-sm font-medium rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                Chat
              </Link>
              <Link
                href="/admin/observability"
                className="px-4 py-2 text-sm font-medium rounded-lg bg-hiplink-blue text-white"
              >
                Observability
              </Link>
              <Link
                href="/admin/evaluations"
                className="px-4 py-2 text-sm font-medium rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 transition-colors"
              >
                Evaluations
              </Link>
              <Link
                href="/documents"
                className="px-4 py-2 text-sm font-medium rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 transition-colors"
              >
                Documents
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 max-w-6xl mx-auto w-full px-4 py-6">
        {children}
      </div>
    </div>
  )
}