'use client'

import { useState, useRef, FormEvent, KeyboardEvent } from 'react'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAuthFetch } from '@/hooks/useApi'
import { useAuth } from '@/contexts/AuthContext'

interface Citation {
  index: number
  source_file_name: string
  content_snippet: string
  relevance_score?: number
  chunk_id?: string
}

interface GroupedSource {
  source_file_name: string
  sections_used: number
  highest_score: number | null
  confidence: 'High' | 'Medium' | 'Low'
  excerpts: string[]
  indices?: number[]
  show_debug_details?: boolean
}

interface Message {
  role: 'user' | 'assistant'
  text: string
  citations?: Citation[]
  grouped_sources?: GroupedSource[]
  debug_info?: Record<string, unknown>
}

type ChatMode = 'general_chat' | 'knowledge_base' | 'debug'

const modeLabels: Record<ChatMode, string> = {
  general_chat: 'General Chat',
  knowledge_base: 'Knowledge Base',
  debug: 'Debug',
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<ChatMode>('general_chat')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const authFetch = useAuthFetch()
  const { devUser, auth } = useAuth()
  const isAdmin = auth?.is_admin ?? false

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
      
      const assistantMessage: Message = {
        role: 'assistant',
        text: data.message,
        citations: data.citations,
        grouped_sources: data.grouped_sources,
        debug_info: data.debug_info,
      }
      
      setMessages((prev) => [...prev, assistantMessage])
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

  // Check if current mode should show sources
  const shouldShowSources = mode !== 'general_chat'

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
        </div>
      </header>

      {/* Mode Selector */}
      <div className="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600 font-medium">Mode:</span>
          {(Object.keys(modeLabels) as ChatMode[]).filter(m => m !== 'debug' || isAdmin === true).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`px-3 py-1 text-sm rounded transition-colors ${
                mode === m
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-200'
              }`}
              title={m === 'debug' ? 'Admin/Developer mode with retrieval details' : undefined}
            >
              {modeLabels[m]}
            </button>
          ))}
        </div>
        <div className="text-xs text-gray-500">
          {mode === 'general_chat' && 'General AI assistant'}
          {mode === 'knowledge_base' && 'Answers from uploaded documents'}
          {mode === 'debug' && 'Retrieval internals (admin)'}
        </div>
      </div>

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
            <p>No messages yet. Start the conversation below.</p>
            <p className="text-sm mt-2">
              {mode === 'general_chat' && 'General Chat: Ask me anything!'}
              {mode === 'knowledge_base' && 'Knowledge Base: I\'ll search your uploaded documents.'}
              {mode === 'debug' && 'Debug Mode: Shows retrieval scores and chunk IDs.'}
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-2 rounded-lg ${
                msg.role === 'user'
                  ? 'bg-blue-500 text-white rounded-br-none'
                  : 'bg-gray-100 text-gray-900 rounded-bl-none'
              }`}>
              {msg.role === 'user' ? (
                <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
              ) : (
                <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.text}
                  </ReactMarkdown>
                </div>
              )}
              
              {/* Sources for Knowledge Base and Debug modes */}
              {msg.role === 'assistant' && shouldShowSources && msg.grouped_sources && msg.grouped_sources.length > 0 && (
                <div className="mt-3 bg-white border border-gray-200 rounded p-3">
                  <p className="font-semibold text-gray-700 mb-2 text-sm">Sources:</p>
                  <div className="space-y-2">
                    {msg.grouped_sources.map((source: GroupedSource, idx: number) => (
                      <SourceCard key={idx} source={source} isDebugMode={mode === 'debug'} />
                    ))}
                  </div>
                </div>
              )}
              
              {/* Debug info for Debug mode */}
              {msg.role === 'assistant' && mode === 'debug' && msg.debug_info && (
                <DebugInfoPanel debugInfo={msg.debug_info} />
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 text-gray-500 px-4 py-2 rounded-lg rounded-bl-none text-sm">
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
          placeholder={`Type your message... (${modeLabels[mode]})`}
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

// Source card component for grouped sources display
function SourceCard({ source, isDebugMode = false }: { source: GroupedSource; isDebugMode?: boolean }) {
  const [expanded, setExpanded] = useState(false)

  const confidenceColors = {
    High: 'bg-green-100 text-green-700 border-green-200',
    Medium: 'bg-yellow-100 text-yellow-700 border-yellow-200',
    Low: 'bg-gray-100 text-gray-600 border-gray-200',
  }

  return (
    <div className="border border-gray-200 rounded-lg bg-white overflow-hidden">
      <div className="p-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h4 className="font-medium text-gray-900 text-sm">{source.source_file_name}</h4>
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              <span className={`px-2 py-0.5 rounded text-xs font-medium border ${confidenceColors[source.confidence]}`}>
                {source.confidence}
              </span>
              <span className="text-gray-500 text-xs">
                {source.sections_used} {source.sections_used === 1 ? 'section' : 'sections'} used
              </span>
              {/* Only show debug details in Debug mode */}
              {isDebugMode && source.highest_score !== null && (
                <span className="text-gray-400 text-xs">
                  (score: {source.highest_score.toFixed(2)})
                </span>
              )}
              {isDebugMode && source.indices && source.indices.length > 0 && (
                <span className="text-gray-400 text-xs">
                  [indices: {source.indices.join(', ')}]
                </span>
              )}
            </div>
          </div>
        </div>
        {source.excerpts.length > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-2 text-blue-600 hover:text-blue-800 text-xs font-medium flex items-center gap-1"
          >
            {expanded ? '▲ Hide excerpts' : '▼ Show excerpts'}
          </button>
        )}
      </div>
      {expanded && source.excerpts.length > 0 && (
        <div className="border-t border-gray-100 p-3 bg-gray-50">
          <div className="space-y-2">
            {source.excerpts.map((excerpt, i) => (
              <p key={i} className="text-gray-600 text-xs leading-relaxed">
                {excerpt}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Debug info panel for developer/admin mode
function DebugInfoPanel({ debugInfo }: { debugInfo: Record<string, unknown> }) {
  const [expanded, setExpanded] = useState(false)
  
  return (
    <div className="mt-3 bg-gray-900 text-gray-100 rounded p-3 text-xs font-mono">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-gray-400 hover:text-gray-200 mb-1"
      >
        {expanded ? '▼' : '▶'} Debug Info
      </button>
      {expanded && (
        <pre className="overflow-x-auto whitespace-pre-wrap break-all">
          {JSON.stringify(debugInfo, null, 2)}
        </pre>
      )}
    </div>
  )
}
