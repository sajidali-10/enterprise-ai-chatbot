'use client'

import { useEffect, useState } from 'react'

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

      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-md">
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
    </main>
  )
}