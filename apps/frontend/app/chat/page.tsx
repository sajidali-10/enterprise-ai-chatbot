'use client'

import { useState, useRef, FormEvent, KeyboardEvent } from 'react'
import Link from 'next/link'
import Image from 'next/image'
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
  observation_id?: number
}

type ChatMode = 'general_chat' | 'knowledge_base' | 'debug'

const modeLabels: Record<ChatMode, string> = {
  general_chat: 'General Chat',
  knowledge_base: 'Knowledge Base',
  debug: 'Debug',
}

const modeHelperText: Record<ChatMode, string> = {
  general_chat: 'Uses general AI conversation',
  knowledge_base: 'Answers only from uploaded documents',
  debug: 'Shows retrieval and grounding details',
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
        observation_id: data.observation_id,
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

  const shouldShowSources = mode !== 'general_chat'
  const isDebugMode = mode === 'debug'

  return (
    <div className="min-h-screen bg-hiplink-background flex flex-col">
      {/* Header */}
      <header className="brand-header flex-shrink-0">
        <div className="max-w-2xl mx-auto px-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Link href="/" className="flex items-center gap-2">
                <Image
                  src="/hiplink-logo.png"
                  alt="HipLink"
                  width={32}
                  height={32}
                  className="object-contain"
                />
                <span className="text-lg font-semibold text-hiplink-dark hidden sm:inline">HipLink AI Assistant</span>
              </Link>
            </div>
            <div className="flex items-center space-x-4">
              <Link
                href="/admin/observability"
                className="text-sm text-hiplink-secondary hover:text-hiplink-blue font-medium transition-colors"
              >
                Observability
              </Link>
              <Link
                href="/auth"
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-full text-sm ${
                  devUser
                    ? 'bg-green-50 text-green-700 hover:bg-green-100'
                    : 'bg-gray-100 text-hiplink-secondary hover:bg-gray-200'
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${devUser ? 'bg-hiplink-success' : 'bg-gray-400'}`} />
                <span>{devUser || 'Guest'}</span>
              </Link>
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 max-w-2xl mx-auto w-full px-4 py-4 flex flex-col">
        {/* Mode Selector */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4 p-3 bg-white border border-hiplink-border rounded-lg">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm text-hiplink-secondary font-medium">Mode:</span>
            {(Object.keys(modeLabels) as ChatMode[]).filter(m => m !== 'debug' || isAdmin === true).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`px-3 py-1.5 text-sm rounded-lg transition-all font-medium ${
                  mode === m
                    ? 'bg-hiplink-blue text-white shadow-sm'
                    : 'bg-gray-100 text-hiplink-dark hover:bg-gray-200'
                }`}
                title={m === 'debug' ? 'Admin/Developer mode with retrieval details' : undefined}
              >
                {modeLabels[m]}
              </button>
            ))}
          </div>
          <div className="text-xs text-hiplink-secondary italic">
            {modeHelperText[mode]}
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="bg-red-50 border border-hiplink-error text-hiplink-error px-4 py-3 rounded-lg mb-4">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Message list */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4">
          {messages.length === 0 && !loading && (
            <div className="text-hiplink-secondary text-center mt-12">
              <div className="mb-4">
                <Image
                  src="/hiplink-logo.png"
                  alt="HipLink"
                  width={64}
                  height={64}
                  className="object-contain mx-auto opacity-50"
                />
              </div>
              <p className="text-lg font-medium mb-2">Welcome to HipLink AI Assistant</p>
              <p className="text-sm">
                {mode === 'general_chat' && 'General Chat: Ask me anything!'}
                {mode === 'knowledge_base' && "Knowledge Base: I'll search your uploaded documents."}
                {mode === 'debug' && 'Debug Mode: Shows retrieval scores and chunk IDs.'}
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[85%] px-4 py-3 rounded-2xl ${
                msg.role === 'user'
                  ? 'bg-hiplink-blue text-white rounded-br-sm'
                  : 'bg-white text-hiplink-dark border border-hiplink-border rounded-bl-sm'
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
                  <div className="mt-4 bg-hiplink-background border border-hiplink-border rounded-lg p-3">
                    <p className="font-semibold text-hiplink-dark mb-2 text-sm">Sources:</p>
                    <div className="space-y-2">
                      {msg.grouped_sources.map((source: GroupedSource, idx: number) => (
                        <SourceCard key={idx} source={source} isDebugMode={isDebugMode} />
                      ))}
                    </div>
                  </div>
                )}
              
                {/* Debug info for Debug mode */}
                {msg.role === 'assistant' && isDebugMode && msg.debug_info && (
                  <DebugInfoPanel debugInfo={msg.debug_info} />
                )}
              
                {/* Feedback buttons for assistant messages */}
                {msg.role === 'assistant' && msg.observation_id && (
                  <FeedbackButtons observationId={msg.observation_id} />
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white text-hiplink-secondary px-4 py-3 rounded-2xl rounded-bl-sm border border-hiplink-border text-sm">
                <span className="inline-flex items-center gap-2">
                  <span className="w-2 h-2 bg-hiplink-blue rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-hiplink-blue rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-hiplink-blue rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  Thinking...
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input row */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            placeholder={`Type your message...`}
            className="flex-1 border border-hiplink-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-hiplink-blue focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed bg-white text-hiplink-dark placeholder:text-gray-400"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="btn-primary px-6 py-3"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}

// Source card component for grouped sources display
function SourceCard({ source, isDebugMode = false }: { source: GroupedSource; isDebugMode?: boolean }) {
  const [expanded, setExpanded] = useState(false)

  const confidenceColors = {
    High: 'bg-green-100 text-hiplink-success border-green-200',
    Medium: 'bg-amber-100 text-hiplink-warning border-amber-200',
    Low: 'bg-gray-100 text-hiplink-secondary border-gray-200',
  }

  return (
    <div className="border border-hiplink-border rounded-lg bg-white overflow-hidden">
      <div className="p-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h4 className="font-medium text-hiplink-dark text-sm">{source.source_file_name}</h4>
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              <span className={`px-2 py-0.5 rounded text-xs font-medium border ${confidenceColors[source.confidence]}`}>
                {source.confidence}
              </span>
              <span className="text-hiplink-secondary text-xs">
                {source.sections_used} {source.sections_used === 1 ? 'section' : 'sections'} used
              </span>
              {/* Show debug details only in Debug mode */}
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
            className="mt-2 text-hiplink-blue hover:text-hiplink-blue-dark text-xs font-medium flex items-center gap-1"
          >
            {expanded ? '▲ Hide excerpts' : '▼ Show excerpts'}
          </button>
        )}
      </div>
      {expanded && source.excerpts.length > 0 && (
        <div className="border-t border-hiplink-border p-3 bg-hiplink-background">
          <div className="space-y-2">
            {source.excerpts.map((excerpt, i) => (
              <p key={i} className="text-hiplink-secondary text-xs leading-relaxed">
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
    <div className="mt-3 bg-gray-900 text-gray-100 rounded-lg p-3 text-xs font-mono">
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

// Feedback buttons component
function FeedbackButtons({ observationId }: { observationId: number }) {
  const [feedbackState, setFeedbackState] = useState<'none' | 'submitted' | 'reason'>('none')
  const [selectedReason, setSelectedReason] = useState<string>('')
  const [comment, setComment] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const authFetch = useAuthFetch()

  const feedbackReasons = [
    { id: 'wrong_answer', label: 'Wrong answer' },
    { id: 'wrong_source', label: 'Wrong source' },
    { id: 'missing_information', label: 'Missing information' },
    { id: 'too_long', label: 'Too long' },
    { id: 'unclear', label: 'Unclear' },
    { id: 'other', label: 'Other' },
  ]

  const handleFeedback = async (rating: 'helpful' | 'not_helpful') => {
    if (rating === 'not_helpful') {
      setFeedbackState('reason')
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      const res = await authFetch(`/api/chat/${observationId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating }),
      })

      if (!res.ok) {
        throw new Error('Failed to submit feedback')
      }

      setFeedbackState('submitted')
    } catch (err) {
      setError('Failed to submit feedback')
    } finally {
      setSubmitting(false)
    }
  }

  const handleReasonSubmit = async () => {
    setSubmitting(true)
    setError(null)

    try {
      const res = await authFetch(`/api/chat/${observationId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rating: 'not_helpful',
          reason: selectedReason,
          comment: comment || undefined,
        }),
      })

      if (!res.ok) {
        throw new Error('Failed to submit feedback')
      }

      setFeedbackState('submitted')
    } catch (err) {
      setError('Failed to submit feedback')
    } finally {
      setSubmitting(false)
    }
  }

  if (feedbackState === 'submitted') {
    return (
      <div className="mt-3 text-xs text-hiplink-success font-medium bg-green-50 px-3 py-2 rounded-lg">
        ✓ Thank you for your feedback!
      </div>
    )
  }

  if (feedbackState === 'reason') {
    return (
      <div className="mt-3 bg-hiplink-background border border-hiplink-border rounded-lg p-3">
        <p className="text-sm text-hiplink-dark mb-2 font-medium">Why was this not helpful?</p>
        <div className="flex flex-wrap gap-2 mb-3">
          {feedbackReasons.map((reason) => (
            <button
              key={reason.id}
              onClick={() => setSelectedReason(reason.id)}
              className={`px-2 py-1 text-xs rounded-lg ${
                selectedReason === reason.id
                  ? 'bg-hiplink-blue text-white'
                  : 'bg-white text-hiplink-dark border border-hiplink-border hover:bg-gray-50'
              }`}
            >
              {reason.label}
            </button>
          ))}
        </div>
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="Optional comment..."
          className="w-full border border-hiplink-border rounded-lg p-2 text-xs mb-2 resize-none focus:outline-none focus:ring-2 focus:ring-hiplink-blue bg-white"
          rows={2}
        />
        <div className="flex space-x-2">
          <button
            onClick={handleReasonSubmit}
            disabled={!selectedReason || submitting}
            className="btn-primary text-xs py-1.5 px-3"
          >
            {submitting ? 'Submitting...' : 'Submit'}
          </button>
          <button
            onClick={() => setFeedbackState('none')}
            className="btn-secondary text-xs py-1.5 px-3"
          >
            Cancel
          </button>
        </div>
        {error && <p className="text-xs text-hiplink-error mt-1">{error}</p>}
      </div>
    )
  }

  return (
    <div className="mt-3 flex items-center space-x-2">
      <span className="text-xs text-hiplink-secondary">Was this helpful?</span>
      <button
        onClick={() => handleFeedback('helpful')}
        disabled={submitting}
        className="px-2 py-1 text-xs bg-white border border-hiplink-border text-hiplink-dark rounded-lg hover:bg-green-50 hover:text-hiplink-success hover:border-green-200 transition-all"
        title="Helpful"
      >
        👍 Helpful
      </button>
      <button
        onClick={() => handleFeedback('not_helpful')}
        disabled={submitting}
        className="px-2 py-1 text-xs bg-white border border-hiplink-border text-hiplink-dark rounded-lg hover:bg-red-50 hover:text-hiplink-error hover:border-red-200 transition-all"
        title="Not helpful"
      >
        👎 Not helpful
      </button>
      {error && <span className="text-xs text-hiplink-error">{error}</span>}
    </div>
  )
}