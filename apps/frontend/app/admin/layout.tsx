'use client'

import Link from 'next/link'
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
      <div className="max-w-2xl mx-auto h-screen flex flex-col items-center justify-center p-4">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <strong>Access Denied:</strong> Admin access required.
        </div>
        <Link href="/chat" className="mt-4 text-blue-600 hover:text-blue-800">
          &larr; Back to Chat
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto h-screen flex flex-col p-4">
      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Link href="/chat" className="text-blue-600 hover:text-blue-800 text-sm font-medium mr-4">
            &larr; Back to Chat
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">Admin</h1>
        </div>
        <nav className="flex items-center space-x-4">
          <Link
            href="/admin/observability"
            className="px-3 py-1 text-sm rounded bg-blue-500 text-white hover:bg-blue-600"
          >
            Observability
          </Link>
          <Link
            href="/admin/evaluations"
            className="px-3 py-1 text-sm rounded bg-blue-500 text-white hover:bg-blue-600"
          >
            Evaluations
          </Link>
        </nav>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {children}
      </div>
    </div>
  )
}