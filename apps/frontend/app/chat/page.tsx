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
  debug: 'Shows retrieval, grounding, and citation metadata',
}

const examplePrompts = [
  'What are the components of Docker?',
  'Summarize the uploaded document.',
  'Which source supports this answer?',
  'What topics are covered?',
]

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

  const handleExampleClick = (prompt: string) => {
    setInput(prompt)
  }

  const shouldShowSources = mode !== 'general_chat'
  const isDebugMode = mode === 'debug'

  return (
    <div className="min-h-screen bg-hiplink-background flex flex-col">
      {/* Header */}
      <header className="brand-header flex-shrink-0">
        <div className="max-w-4xl mx-auto px-4">
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
                <span className="text-lg font-semibold text-hiplink-dark">HipLink AI Assistant</span>
              </Link>
            </div>
            <nav className="flex items-center gap-1">
              <Link
                href="/chat"
                className="px-3 py-2 text-sm font-medium rounded-lg bg-hiplink-blue text-white"
              >
                Chat
              </Link>
              <Link
                href="/documents"
                className="px-3 py-2 text-sm font-medium rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 transition-colors"
              >
                Documents
              </Link>
              {isAdmin && (
                <>
                  <Link
                    href="/admin/observability"
                    className="px-3 py-2 text-sm font-medium rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 transition-colors"
                  >
                    Observability
                  </Link>
                  <Link
                    href="/admin/evaluations"
                    className="px-3 py-2 text-sm font-medium rounded-lg text-hiplink-secondary hover:text-hiplink-blue hover:bg-blue-50 transition-colors"
                  >
                    Evaluations
                  </Link>
                </>
              )}
              <Link
                href="/auth"
                className={`ml-2 flex items-center space-x-2 px-3 py-2 rounded-lg text-sm ${
                  devUser
                    ? 'bg-green-50 text-green-700'
                    : 'bg-gray-100 text-hiplink-secondary'
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${devUser ? 'bg-hiplink-success' : 'bg-gray-400'}`} />
                <span>{devUser || 'Guest'}</span>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-6 flex flex-col">
        {/* Mode Selector - Segmented Control */}
        <div className="mb-6 p-4 bg-white border border-hiplink-border rounded-xl shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-1 bg-gray-100 p-1 rounded-lg">
              {(Object.keys(modeLabels) as ChatMode[]).filter(m => m !== 'debug' || isAdmin === true).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                    mode === m
                      ? 'bg-hiplink-blue text-white shadow-sm'
                      : 'bg-transparent text-hiplink-secondary hover:text-hiplink-dark'
                  }`}
                  title={m === 'debug' ? 'Admin/Developer mode with retrieval details' : undefined}
                >
                  {modeLabels[m]}
                </button>
              ))}
            </div>
            <p className="text-sm text-hiplink-secondary italic">
              {modeHelperText[mode]}
            </p>
          </div>
        </div>

        {/* Chat Container */}
        <div className="flex-1 flex flex-col bg-white border border-hiplink-border rounded-xl shadow-sm overflow-hidden">
          {/* Error banner */}
          {error && (
            <div className="mx-4 mt-4 bg-red-50 border border-hiplink-error text-hiplink-error px-4 py-3 rounded-lg">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Message list */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && !loading && (
              <WelcomeState 
                mode={mode} 
                onExampleClick={handleExampleClick} 
                examplePrompts={examplePrompts}
              />
            )}

            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[80%] ${
                  msg.role === 'user'
                    ? 'bg-hiplink-blue text-white'
                    : 'bg-hiplink-background text-hiplink-dark'
                } px-5 py-3 rounded-2xl ${
                  msg.role === 'user'
                    ? 'rounded-br-sm'
                    : 'rounded-bl-sm border border-hiplink-border'
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
                    <div className="mt-4 pt-4 border-t border-hiplink-border">
                      <p className="font-semibold text-hiplink-dark mb-3 text-sm flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Sources
                      </p>
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
                <div className="bg-hiplink-background text-hiplink-secondary px-5 py-3 rounded-2xl rounded-bl-sm border border-hiplink-border">
                  <span className="inline-flex items-center gap-2">
                    <span className="w-2 h-2 bg-hiplink-blue rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-hiplink-blue rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-hiplink-blue rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    <span className="ml-1">Thinking...</span>
                  </span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input row - sticky at bottom */}
          <div className="border-t border-hiplink-border p-4 bg-white">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                placeholder="Type your message..."
                className="flex-1 border border-hiplink-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-hiplink-blue focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed bg-white text-hiplink-dark placeholder:text-gray-400"
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="btn-primary px-6 py-3 flex items-center gap-2"
              >
                <span>Send</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

// Welcome state component
function WelcomeState({ 
  mode, 
  onExampleClick, 
  examplePrompts 
}: { 
  mode: ChatMode
  onExampleClick: (prompt: string) => void
  examplePrompts: string[]
}) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <div className="mb-6">
        <Image
          src="/hiplink-logo.png"
          alt="HipLink"
          width={80}
          height={80}
          className="object-contain"
        />
      </div>
      <h2 className="text-2xl font-bold text-hiplink-dark mb-2">Welcome to HipLink AI Assistant</h2>
      <p className="text-hiplink-secondary mb-8 max-w-md">
        {mode === 'general_chat' 
          ? 'Ask me anything! I\'ll use general AI conversation to help you.'
          : mode === 'knowledge_base'
          ? 'I\'ll search through your uploaded documents to find answers.'
          : 'Debug mode shows retrieval, grounding, and citation metadata.'}
      </p>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
        {examplePrompts.map((prompt, i) => (
          <button
            key={i}
            onClick={() => onExampleClick(prompt)}
            className="text-left px-4 py-3 bg-hiplink-background border border-hiplink-border rounded-lg text-sm text-hiplink-dark hover:border-hiplink-blue hover:bg-blue-50 transition-colors"
          >
            {prompt}
          </button>
        ))}
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
    <div className="border border-hiplink-border rounded-lg bg-white overflow-hidden hover:border-gray-300 transition-colors">
      <div className="p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-hiplink-dark text-sm truncate">{source.source_file_name}</h4>
            <div className="flex items-center gap-2 mt-1.5 flex-wrap">
              <span className={`px-2 py-0.5 rounded text-xs font-medium border ${confidenceColors[source.confidence]}`}>
                {source.confidence}
              </span>
              <span className="text-hiplink-secondary text-xs">
                {source.sections_used} {source.sections_used === 1 ? 'section' : 'sections'} used
              </span>
              {/* Show debug details only in Debug mode */}
              {isDebugMode && source.highest_score !== null && (
                <span className="text-gray-400 text-xs font-mono">
                  score: {source.highest_score.toFixed(3)}
                </span>
              )}
              {isDebugMode && source.indices && source.indices.length > 0 && (
                <span className="text-gray-400 text-xs font-mono">
                  [{source.indices.join(', ')}]
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
            {expanded ? (
              <>
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                </svg>
                Hide excerpts
              </>
            ) : (
              <>
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
                Show excerpts
              </>
            )}
          </button>
        )}
      </div>
      {expanded && source.excerpts.length > 0 && (
        <div className="border-t border-hiplink-border px-3 py-3 bg-gray-50">
          <div className="space-y-2">
            {source.excerpts.map((excerpt, i) => (
              <p key={i} className="text-hiplink-secondary text-xs leading-relaxed bg-white px-2 py-1.5 rounded border border-gray-100">
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
    <div className="mt-4 bg-gray-900 text-gray-100 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-gray-200 w-full"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={expanded ? "M19 9l-7 7-7-7" : "M9 5l7 7-7 7"} />
        </svg>
        Debug Info
      </button>
      {expanded && (
        <pre className="px-3 pb-3 text-xs font-mono overflow-x-auto whitespace-pre-wrap break-all">
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
    } catch {
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
    } catch {
      setError('Failed to submit feedback')
    } finally {
      setSubmitting(false)
    }
  }

  if (feedbackState === 'submitted') {
    return (
      <div className="mt-4 text-sm text-hiplink-success font-medium flex items-center gap-2">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
        Thank you for your feedback!
      </div>
    )
  }

  if (feedbackState === 'reason') {
    return (
      <div className="mt-4 bg-hiplink-background border border-hiplink-border rounded-lg p-4">
        <p className="text-sm text-hiplink-dark mb-3 font-medium">Why was this not helpful?</p>
        <div className="flex flex-wrap gap-2 mb-3">
          {feedbackReasons.map((reason) => (
            <button
              key={reason.id}
              onClick={() => setSelectedReason(reason.id)}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
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
          className="w-full border border-hiplink-border rounded-lg p-2 text-sm mb-3 resize-none focus:outline-none focus:ring-2 focus:ring-hiplink-blue bg-white"
          rows={2}
        />
        <div className="flex gap-2">
          <button
            onClick={handleReasonSubmit}
            disabled={!selectedReason || submitting}
            className="btn-primary text-sm py-2 px-4"
          >
            {submitting ? 'Submitting...' : 'Submit'}
          </button>
          <button
            onClick={() => setFeedbackState('none')}
            className="btn-secondary text-sm py-2 px-4"
          >
            Cancel
          </button>
        </div>
        {error && <p className="text-xs text-hiplink-error mt-2">{error}</p>}
      </div>
    )
  }

  return (
    <div className="mt-4 pt-3 border-t border-hiplink-border flex items-center gap-3">
      <span className="text-xs text-hiplink-secondary">Was this helpful?</span>
      <div className="flex gap-1">
        <button
          onClick={() => handleFeedback('helpful')}
          disabled={submitting}
          className="px-3 py-1.5 text-sm bg-white border border-hiplink-border text-hiplink-dark rounded-lg hover:bg-green-50 hover:text-hiplink-success hover:border-green-200 transition-all flex items-center gap-1"
          title="Helpful"
        >
          <span>👍</span>
        </button>
        <button
          onClick={() => handleFeedback('not_helpful')}
          disabled={submitting}
          className="px-3 py-1.5 text-sm bg-white border border-hiplink-border text-hiplink-dark rounded-lg hover:bg-red-50 hover:text-hiplink-error hover:border-red-200 transition-all flex items-center gap-1"
          title="Not helpful"
        >
          <span>👎</span>
        </button>
      </div>
      {error && <span className="text-xs text-hiplink-error">{error}</span>}
    </div>
  )
}