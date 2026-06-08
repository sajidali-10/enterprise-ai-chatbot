'use client'

import { useState, useRef, FormEvent, KeyboardEvent } from 'react'
import Link from 'next/link'
import { useAuthFetch } from '@/hooks/useApi'
import { useAuth } from '@/contexts/AuthContext'

interface Citation {
  index: number
  source_file_name: string
  content_snippet: string
  relevance_score?: number
}

interface Message {
  role: 'user' | 'assistant'
  text: string
  citations?: Citation[]
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'normal' | 'rag'>('normal')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const authFetch = useAuthFetch()
  const { devUser } = useAuth()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSubmit = async (e?: FormEvent) => {
    e?.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || loading) return

    const userMessage: Message = { role: 'user', text: trimmed }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setError(null)
    scrollToBottom()

    try {
      const res = await authFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, mode }),
      })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.message, citations: data.citations },
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
      scrollToBottom()
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="max-w-2xl mx-auto h-screen flex flex-col p-4">
      {/* Header */}
      <header className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <Link
            href="/"
            className="text-blue-600 hover:text-blue-800 text-sm font-medium mr-4"
          >
            &larr; Back
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">Chat</h1>
        </div>
        <div className="flex items-center space-x-4">
          {/* Auth Status */}
          <Link
            href="/auth"
            className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
              devUser
                ? 'bg-green-100 text-green-700 hover:bg-green-200'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${devUser ? 'bg-green-500' : 'bg-gray-400'}`} />
            <span>{devUser || 'Not authenticated'}</span>
          </Link>
          <span className="text-sm text-gray-600">Mode:</span>
          <button
            type="button"
            onClick={() => setMode('normal')}
            className={`px-3 py-1 text-sm rounded ${
              mode === 'normal'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Normal
          </button>
          <button
            type="button"
            onClick={() => setMode('rag')}
            className={`px-3 py-1 text-sm rounded ${
              mode === 'rag'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            RAG
          </button>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Message list */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.length === 0 && !loading && (
          <div className="text-gray-400 text-center mt-8">
            No messages yet. Start the conversation below.
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`max-w-xs md:max-w-md px-4 py-2 rounded-lg text-sm ${
                msg.role === 'user'
                  ? 'bg-blue-500 text-white rounded-br-none'
                  : 'bg-gray-200 text-gray-900 rounded-bl-none'
              }`}>
              {msg.text}
              {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 bg-gray-50 border border-gray-200 rounded p-3 text-sm">
                  <p className="font-semibold mb-2">Sources:</p>
                  {msg.citations.map((cit: Citation) => (
                    <div key={cit.index} className="mb-2">
                      <span className="font-medium text-blue-600">[{cit.index}] {cit.source_file_name}</span>
                      <span className="text-gray-500 ml-2">(score: {cit.relevance_score?.toFixed(2)})</span>
                      <p className="text-gray-700 mt-1">{cit.content_snippet}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-200 text-gray-500 px-4 py-2 rounded-lg rounded-bl-none text-sm">
              Thinking...
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input row */}
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          placeholder="Type your message..."
          className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-blue-500 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-600 disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  )
}