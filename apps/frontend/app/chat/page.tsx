'use client'

import { useState, useRef, FormEvent, KeyboardEvent } from 'react'
import Image from 'next/image'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getPermissions, getDefaultChatMode, normalizePermissions, type UserRole, type PermissionFlags } from '@/lib/permissions'
import { useAuthFetch } from '@/hooks/useApi'
import { useAuth } from '@/contexts/AuthContext'
import { useTheme } from '@/contexts/ThemeContext'
import Header from '@/components/Header'

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

interface ChatModeConfig {
  id: ChatMode
  label: string
  helperText: string
  permission?: 'canUseGeneralChat' | 'canUseKnowledgeBase' | 'canUseDebug'
}

const allModes: ChatModeConfig[] = [
  { id: 'general_chat', label: 'General Chat', helperText: 'Uses general AI conversation', permission: 'canUseGeneralChat' },
  { id: 'knowledge_base', label: 'Knowledge Base', helperText: 'Answers only from uploaded documents', permission: 'canUseKnowledgeBase' },
  { id: 'debug', label: 'Debug', helperText: 'Shows retrieval, grounding, and citation metadata', permission: 'canUseDebug' },
]

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
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const authFetch = useAuthFetch()
  const { devUser, auth } = useAuth()
  const { theme } = useTheme()
  
  // Determine effective role and permissions
  const rawRole = auth?.role || 
    (devUser === 'admin_user' ? 'admin' : 
     devUser === 'regular_user' ? 'user' : 
     devUser === 'viewer_user' ? 'viewer' : undefined)
  const perms: PermissionFlags = normalizePermissions(auth?.permissions, rawRole)

  // Set default mode based on permissions (viewer only gets KB)
  const [mode, setMode] = useState<ChatMode>(() => getDefaultChatMode(rawRole) as ChatMode)

  // Filter available modes based on permissions
  const availableModes = allModes.filter(m => !m.permission || perms[m.permission])
  
  // Enforce permission check when mode changes
  const handleModeChange = (newMode: ChatMode) => {
    const config = allModes.find(m => m.id === newMode)
    if (config?.permission && !perms[config.permission]) {
      // User doesn't have permission - switch to default allowed mode
      setMode(getDefaultChatMode(rawRole) as ChatMode)
      return
    }
    setMode(newMode)
  }

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
  const isViewerMode = rawRole === 'viewer'

  return (
    <div className="min-h-screen bg-hiplink-background dark:bg-dark-bg flex flex-col">
      <Header showAdminNav />
      
      {/* Main Content */}
      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-6 flex flex-col">
        {/* Mode Selector - Segmented Control */}
        <div className="mb-6 p-4 bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-1 bg-gray-100 dark:bg-dark-elevated p-1 rounded-lg">
              {availableModes.map((modeConfig) => (
                <button
                  key={modeConfig.id}
                  type="button"
                  onClick={() => handleModeChange(modeConfig.id)}
                  className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                    mode === modeConfig.id
                      ? 'bg-hiplink-blue text-white shadow-sm'
                      : 'bg-transparent text-hiplink-secondary dark:text-dark-text-muted hover:text-hiplink-dark dark:hover:text-dark-text'
                  }`}
                >
                  {modeConfig.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              {isViewerMode ? (
                <span className="text-xs text-hiplink-secondary dark:text-dark-text-dim italic bg-blue-50 dark:bg-sky-900/20 px-2 py-1 rounded">
                  Viewer mode: read-only answers from uploaded knowledge sources.
                </span>
              ) : (
                <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim italic">
                  {availableModes.find(m => m.id === mode)?.helperText}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Chat Container */}
        <div className="flex-1 flex flex-col bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-xl shadow-sm overflow-hidden">
          {/* Error banner */}
          {error && (
            <div className="mx-4 mt-4 bg-red-50 dark:bg-red-900/20 border border-hiplink-error text-hiplink-error dark:text-red-400 px-4 py-3 rounded-lg">
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
                    ? 'bg-hiplink-blue dark:bg-sky-600 text-white'
                    : 'bg-hiplink-background dark:bg-dark-elevated text-hiplink-dark dark:text-dark-text'
                } px-5 py-3 rounded-2xl ${
                  msg.role === 'user'
                    ? 'rounded-br-sm'
                    : 'rounded-bl-sm border border-hiplink-border dark:border-dark-border'
                }`}>
                  {msg.role === 'user' ? (
                    <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
                  ) : (
                    <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 dark:prose-invert">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.text}
                      </ReactMarkdown>
                    </div>
                  )}
                
                  {/* Sources for Knowledge Base and Debug modes */}
                  {msg.role === 'assistant' && shouldShowSources && msg.grouped_sources && msg.grouped_sources.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-hiplink-border dark:border-dark-border">
                      <p className="font-semibold text-hiplink-dark dark:text-dark-text mb-3 text-sm flex items-center gap-2">
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
                <div className="bg-hiplink-background dark:bg-dark-elevated text-hiplink-secondary dark:text-dark-text-muted px-5 py-3 rounded-2xl rounded-bl-sm border border-hiplink-border dark:border-dark-border">
                  <span className="inline-flex items-center gap-2">
                    <span className="w-2 h-2 bg-hiplink-blue dark:bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-hiplink-blue dark:bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-hiplink-blue dark:bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    <span className="ml-1">Thinking...</span>
                  </span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input row - sticky at bottom */}
          <div className="border-t border-hiplink-border dark:border-dark-border p-4 bg-white dark:bg-dark-card">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                placeholder="Type your message..."
                className="flex-1 border border-hiplink-border dark:border-dark-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-hiplink-blue dark:focus:ring-sky-400 focus:border-transparent disabled:bg-gray-100 dark:bg-dark-elevated disabled:cursor-not-allowed text-hiplink-dark dark:text-dark-text placeholder:text-gray-400 dark:placeholder:text-dark-text-dim bg-white dark:bg-dark-card"
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
      <h2 className="text-2xl font-bold text-hiplink-dark dark:text-dark-text mb-2">Welcome to HipLink AI Assistant</h2>
      <p className="text-hiplink-secondary dark:text-dark-text-muted mb-8 max-w-md">
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
            className="text-left px-4 py-3 bg-hiplink-background dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border rounded-lg text-sm text-hiplink-dark dark:text-dark-text hover:border-hiplink-blue dark:hover:border-sky-400 hover:bg-blue-50 dark:hover:bg-sky-900/20 transition-colors"
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
    High: 'bg-green-100 dark:bg-green-900/30 text-hiplink-success dark:text-green-400 border-green-200 dark:border-green-800',
    Medium: 'bg-amber-100 dark:bg-amber-900/30 text-hiplink-warning dark:text-amber-400 border-amber-200 dark:border-amber-800',
    Low: 'bg-gray-100 dark:bg-gray-800 text-hiplink-secondary dark:text-dark-text-dim border-gray-200 dark:border-gray-700',
  }

  return (
    <div className="border border-hiplink-border dark:border-dark-border rounded-lg bg-white dark:bg-dark-card overflow-hidden hover:border-gray-300 dark:hover:border-slate-600 transition-colors">
      <div className="p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-hiplink-dark dark:text-dark-text text-sm truncate">{source.source_file_name}</h4>
            <div className="flex items-center gap-2 mt-1.5 flex-wrap">
              <span className={`px-2 py-0.5 rounded text-xs font-medium border ${confidenceColors[source.confidence]}`}>
                {source.confidence}
              </span>
              <span className="text-hiplink-secondary dark:text-dark-text-dim text-xs">
                {source.sections_used} {source.sections_used === 1 ? 'section' : 'sections'} used
              </span>
              {/* Show debug details only in Debug mode */}
              {isDebugMode && source.highest_score !== null && (
                <span className="text-gray-400 dark:text-dark-text-dim text-xs font-mono">
                  score: {source.highest_score.toFixed(3)}
                </span>
              )}
              {isDebugMode && source.indices && source.indices.length > 0 && (
                <span className="text-gray-400 dark:text-dark-text-dim text-xs font-mono">
                  [{source.indices.join(', ')}]
                </span>
              )}
            </div>
          </div>
        </div>
        {source.excerpts.length > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-2 text-hiplink-blue dark:text-sky-400 hover:text-hiplink-blue-dark dark:hover:text-sky-300 text-xs font-medium flex items-center gap-1"
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
        <div className="border-t border-hiplink-border dark:border-dark-border px-3 py-3 bg-gray-50 dark:bg-dark-elevated">
          <div className="space-y-2">
            {source.excerpts.map((excerpt, i) => (
              <p key={i} className="text-hiplink-secondary dark:text-dark-text-muted text-xs leading-relaxed bg-white dark:bg-dark-card px-2 py-1.5 rounded border border-gray-100 dark:border-dark-border">
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
    <div className="mt-4 bg-gray-900 dark:bg-slate-800 text-gray-100 rounded-lg overflow-hidden">
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
      <div className="mt-4 text-sm text-hiplink-success dark:text-green-400 font-medium flex items-center gap-2">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
        Thank you for your feedback!
      </div>
    )
  }

  if (feedbackState === 'reason') {
    return (
      <div className="mt-4 bg-hiplink-background dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border rounded-lg p-4">
        <p className="text-sm text-hiplink-dark dark:text-dark-text mb-3 font-medium">Why was this not helpful?</p>
        <div className="flex flex-wrap gap-2 mb-3">
          {feedbackReasons.map((reason) => (
            <button
              key={reason.id}
              onClick={() => setSelectedReason(reason.id)}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                selectedReason === reason.id
                  ? 'bg-hiplink-blue dark:bg-sky-600 text-white'
                  : 'bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text border border-hiplink-border dark:border-dark-border hover:bg-gray-50 dark:hover:bg-dark-bg'
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
          className="w-full border border-hiplink-border dark:border-dark-border rounded-lg p-2 text-sm mb-3 resize-none focus:outline-none focus:ring-2 focus:ring-hiplink-blue dark:focus:ring-sky-400 bg-white dark:bg-dark-card text-hiplink-dark dark:text-dark-text"
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
        {error && <p className="text-xs text-hiplink-error dark:text-red-400 mt-2">{error}</p>}
      </div>
    )
  }

  return (
    <div className="mt-4 pt-3 border-t border-hiplink-border dark:border-dark-border flex items-center gap-3">
      <span className="text-xs text-hiplink-secondary dark:text-dark-text-dim">Was this helpful?</span>
      <div className="flex gap-1">
        <button
          onClick={() => handleFeedback('helpful')}
          disabled={submitting}
          className="px-3 py-1.5 text-sm bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border text-hiplink-dark dark:text-dark-text rounded-lg hover:bg-green-50 dark:hover:bg-green-900/30 hover:text-hiplink-success dark:hover:text-green-400 hover:border-green-200 dark:hover:border-green-700 transition-all flex items-center gap-1"
          title="Helpful"
        >
          <span>👍</span>
        </button>
        <button
          onClick={() => handleFeedback('not_helpful')}
          disabled={submitting}
          className="px-3 py-1.5 text-sm bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border text-hiplink-dark dark:text-dark-text rounded-lg hover:bg-red-50 dark:hover:bg-red-900/30 hover:text-hiplink-error dark:hover:text-red-400 hover:border-red-200 dark:hover:border-red-700 transition-all flex items-center gap-1"
          title="Not helpful"
        >
          <span>👎</span>
        </button>
      </div>
      {error && <span className="text-xs text-hiplink-error dark:text-red-400">{error}</span>}
    </div>
  )
}