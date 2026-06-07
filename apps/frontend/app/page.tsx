'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'

export default function Home() {
  const [health, setHealth] = useState<{ status: string; service: string } | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => setHealth(data))
      .catch((err) => setError(err.message))
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold mb-4">Enterprise AI Chatbot</h1>
      <p className="text-lg text-gray-600 mb-8">Phase 0 Foundation</p>

      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-md mb-4">
        <h2 className="text-xl font-semibold mb-4">Backend Health</h2>
        {health ? (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span>Status: {health.status}</span>
          </div>
        ) : error ? (
          <div className="text-red-600">Error: {error}</div>
        ) : (
          <div className="text-gray-500">Checking health...</div>
        )}
      </div>

      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-md">
        <h2 className="text-xl font-semibold mb-4">Phase 1 — Chat Interface</h2>
        <p className="text-gray-600 mb-4">
          Try the end-to-end chat with the backend LLM integration.
        </p>
        <Link
          href="/chat"
          className="inline-block bg-blue-500 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-600 transition-colors"
        >
          Go to Chat
        </Link>
      </div>

      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-md">
        <h2 className="text-xl font-semibold mb-4">Phase 2 — Document Upload</h2>
        <p className="text-gray-600 mb-4">
          Upload and manage documents for future RAG indexing.
        </p>
        <Link
          href="/documents/upload"
          className="inline-block bg-green-500 text-white px-6 py-2 rounded-lg font-medium hover:bg-green-600 transition-colors mr-2"
        >
          Upload Documents
        </Link>
        <Link
          href="/documents"
          className="inline-block text-green-600 hover:text-green-700 font-medium"
        >
          View All
        </Link>
      </div>
    </main>
  )
}