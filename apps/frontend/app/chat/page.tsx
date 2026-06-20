'use client'

import { useState, useRef, FormEvent, KeyboardEvent, useEffect, useCallback } from 'react'
import Image from 'next/image'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getPermissions, getDefaultChatMode, normalizePermissions, type UserRole, type PermissionFlags } from '@/lib/permissions'
import { useAuthFetch } from '@/hooks/useApi'
import { useAuth } from '@/contexts/AuthContext'
import { useTheme } from '@/contexts/ThemeContext'
import Header from '@/components/Header'
import ProtectedRoute from '@/components/ProtectedRoute'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
  suggested_followups?: Suggestion[]
}

// Phase 20B UX Fix: Typed suggestion interface
// Phase 20C: Normalizer added to handle old session data safely
type SuggestionType = 'question' | 'frontend_action' | 'contextual_action'

interface Suggestion {
  label: string
  prompt: string
  type: SuggestionType
}

// Phase 20C: Normalize suggested_followups from any format to safe typed array
function normalizeSuggestedFollowups(value: unknown): Suggestion[] {
  // Handle missing/null/undefined
  if (value === null || value === undefined) {
    return []
  }

  // Handle non-array values
  if (!Array.isArray(value)) {
    return []
  }

  // Normalize each item to typed Suggestion
  const normalized: Suggestion[] = []
  for (const item of value) {
    // Handle string items (old format: string[])
    if (typeof item === 'string') {
      normalized.push({
        label: item,
        prompt: item,
        type: 'question',
      })
      continue
    }

    // Handle object items
    if (item && typeof item === 'object') {
      const obj = item as Record<string, unknown>
      const label = typeof obj.label === 'string' && obj.label.length > 0
        ? obj.label
        : typeof obj.prompt === 'string' && obj.prompt.length > 0
          ? obj.prompt
          : null

      if (label !== null) {
        const type = (obj.type as SuggestionType) || 'question'
        // Validate type is one of the allowed values
        const safeType: SuggestionType = ['question', 'frontend_action', 'contextual_action'].includes(type)
          ? type
          : 'question'

        normalized.push({
          label,
          prompt: typeof obj.prompt === 'string' ? obj.prompt : label,
          type: safeType,
        })
      }
    }
  }

  return normalized
}

interface ChatSession {
  id: number
  user_id: number
  title: string
  mode: 'general' | 'rag'
  created_at: string
  updated_at: string
  archived_at: string | null
  message_count: number
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

// ---------------------------------------------------------------------------
// Page wrapper
// ---------------------------------------------------------------------------

export default function ChatPage() {
  return (
    <ProtectedRoute>
      <ChatPageInner />
    </ProtectedRoute>
  )
}

// ---------------------------------------------------------------------------
// Main chat page with sidebar
// ---------------------------------------------------------------------------

function ChatPageInner() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const authFetch = useAuthFetch()
  const { devUser, auth } = useAuth()
  const { theme } = useTheme()
  const sidebarRef = useRef<HTMLDivElement>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

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
      setMode(getDefaultChatMode(rawRole) as ChatMode)
      return
    }
    setMode(newMode)
  }

  // ---------------------------------------------------------------------------
  // Load sessions from backend
  // ---------------------------------------------------------------------------

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true)
    try {
      const res = await authFetch('/api/chat/sessions')
      if (res.ok) {
        const data = await res.json()
        setSessions(data.sessions || [])
      }
    } catch {
      // Silently fail — sessions load is non-critical
    } finally {
      setSessionsLoading(false)
    }
  }, [authFetch])

  // Load sessions on mount
  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  // ---------------------------------------------------------------------------
  // Load a specific session's messages
  // ---------------------------------------------------------------------------

  const loadSession = useCallback(async (sessionId: number) => {
    try {
      const res = await authFetch(`/api/chat/sessions/${sessionId}`)
      if (res.ok) {
        const data = await res.json()
        // Convert stored messages to UI format
        // Phase 20C: Added defensive handling for missing/undefined fields
        const loadedMessages: Message[] = data.messages.map((m: {
          role: string
          content: string
          citations_json?: string
          retrieved_documents_json?: string
          debug_info?: Record<string, unknown>
          observation_id?: number
          suggested_followups?: unknown  // Phase 20C: may be old format
        }) => {
          // Safely parse citations
          let citations: Citation[] | undefined = undefined
          if (m.citations_json) {
            const parsed = safeParseJson(m.citations_json)
            if (Array.isArray(parsed)) {
              citations = parsed as Citation[]
            }
          }

          // Safely parse grouped sources
          let groupedSources: GroupedSource[] | undefined = undefined
          if (m.retrieved_documents_json) {
            const parsed = safeParseGroupedSources(m.retrieved_documents_json)
            if (Array.isArray(parsed)) {
              groupedSources = parsed as GroupedSource[]
            }
          }

          // Phase 20C: Normalize suggested_followups from any format
          const suggestedFollowups = normalizeSuggestedFollowups(m.suggested_followups)

          return {
            role: (m.role === 'user' || m.role === 'assistant') ? m.role : 'assistant',
            text: typeof m.content === 'string' ? m.content : String(m.content || ''),
            citations,
            grouped_sources: groupedSources,
            debug_info: m.debug_info && typeof m.debug_info === 'object' ? m.debug_info : undefined,
            observation_id: typeof m.observation_id === 'number' ? m.observation_id : undefined,
            suggested_followups: suggestedFollowups.length > 0 ? suggestedFollowups : undefined,
          }
        })
        setMessages(loadedMessages)
        // Update session title in case it changed
        setSessions(prev => prev.map(s =>
          s.id === sessionId ? { ...s, ...data.session } : s
        ))
      } else if (res.status === 404) {
        // Session was deleted — remove from list
        setSessions(prev => prev.filter(s => s.id !== sessionId))
        if (activeSessionId === sessionId) {
          setActiveSessionId(null)
          setMessages([])
        }
      }
    } catch {
      // Silently fail
    }
  }, [authFetch, activeSessionId])

  // ---------------------------------------------------------------------------
  // Start a new chat (clear active session)
  // ---------------------------------------------------------------------------

  const startNewChat = () => {
    setActiveSessionId(null)
    setMessages([])
    setError(null)
    setSidebarOpen(false)
  }

  // ---------------------------------------------------------------------------
  // Select a session
  // ---------------------------------------------------------------------------

  const selectSession = async (sessionId: number) => {
    setActiveSessionId(sessionId)
    setError(null)
    setSidebarOpen(false)
    await loadSession(sessionId)
  }

  // ---------------------------------------------------------------------------
  // Delete/archive a session
  // ---------------------------------------------------------------------------

  const deleteSession = async (sessionId: number) => {
    try {
      const res = await authFetch(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' })
      if (res.ok || res.status === 204) {
        setSessions(prev => prev.filter(s => s.id !== sessionId))
        if (activeSessionId === sessionId) {
          setActiveSessionId(null)
          setMessages([])
        }
      }
    } catch {
      setError('Failed to delete session')
    }
  }

  const archiveSession = async (sessionId: number) => {
    try {
      const res = await authFetch(`/api/chat/sessions/${sessionId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ archived: true }),
      })
      if (res.ok) {
        setSessions(prev => prev.filter(s => s.id !== sessionId))
        if (activeSessionId === sessionId) {
          setActiveSessionId(null)
          setMessages([])
        }
      }
    } catch {
      setError('Failed to archive session')
    }
  }

  // ---------------------------------------------------------------------------
  // Scroll
  // ---------------------------------------------------------------------------

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // ---------------------------------------------------------------------------
  // Submit message
  // ---------------------------------------------------------------------------

  const handleSubmit = async (e?: FormEvent, messageOverride?: string) => {
    e?.preventDefault()
    const trimmed = messageOverride ?? input.trim()
    if (!trimmed || loading) return

    const userMessage: Message = { role: 'user', text: trimmed }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setError(null)
    scrollToBottom()

    try {
      const body: Record<string, unknown> = { message: trimmed, mode }
      if (activeSessionId !== null) {
        body.session_id = activeSessionId
      }

      const res = await authFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }

      const data = await res.json()

      // If backend created a new session, switch to it
      if (data.session_id && data.session_id !== activeSessionId) {
        setActiveSessionId(data.session_id)
        // Reload sessions to pick up the new session
        await loadSessions()
      }

      // Phase 20C: Normalize suggested_followups from any format (including old stored data)
      const normalizedFollowups = normalizeSuggestedFollowups(data.suggested_followups)

      const assistantMessage: Message = {
        role: 'assistant',
        text: data.message,
        citations: data.citations,
        grouped_sources: data.grouped_sources,
        debug_info: data.debug_info,
        observation_id: data.observation_id,
        suggested_followups: normalizedFollowups.length > 0 ? normalizedFollowups : undefined,
      }

      setMessages(prev => [...prev, assistantMessage])
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

  // Phase 20B UX Fix: Handle typed suggestion clicks
  // Phase 20C: contextual_action types are now supported and sent to API like question types
  const handleSuggestionSelect = (suggestion: Suggestion) => {
    if (loading) return

    // Phase 20B UX Fix: Handle frontend_action types without API call
    if (suggestion.type === 'frontend_action') {
      handleFrontendAction(suggestion.label)
      return
    }

    // For 'question' and 'contextual_action' types, send to API
    if ((suggestion.type === 'question' || suggestion.type === 'contextual_action') && suggestion.prompt) {
      const fakeEvent = { preventDefault: () => {} } as FormEvent
      handleSubmit(fakeEvent, suggestion.prompt)
    }
  }

  // Phase 20B UX Fix: Handle frontend actions without API calls
  const handleFrontendAction = (label: string) => {
    const lowerLabel = label.toLowerCase()

    if (lowerLabel.includes('show cited sources') || lowerLabel.includes('show sources')) {
      // Scroll to sources section
      const sourcesEl = document.getElementById('sources-section')
      if (sourcesEl) {
        sourcesEl.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    } else if (lowerLabel.includes('show documents') || lowerLabel.includes('my documents')) {
      // Navigate to Documents page
      window.location.href = '/documents'
    } else if (lowerLabel.includes('upload')) {
      // Navigate to Documents page (upload section)
      window.location.href = '/documents'
    } else if (lowerLabel.includes('rephrase')) {
      // Show a tooltip or do nothing
      // User should manually rephrase
    }
  }

  const shouldShowSources = mode !== 'general_chat'
  const isDebugMode = mode === 'debug'
  const isViewerMode = rawRole === 'viewer'

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-hiplink-background dark:bg-dark-bg flex flex-col">
      <Header showAdminNav />

      <div className="flex-1 flex overflow-hidden">
        {/* Mobile sidebar toggle */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="lg:hidden fixed top-16 left-4 z-30 p-2 bg-white dark:bg-dark-card border border-hiplink-border dark:border-dark-border rounded-lg shadow-md"
          aria-label="Toggle chat history"
        >
          <svg className="w-5 h-5 text-hiplink-secondary dark:text-dark-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        {/* Sidebar */}
        <aside
          ref={sidebarRef}
          className={`
            fixed lg:static inset-y-0 left-0 z-40 w-72 bg-white dark:bg-dark-card border-r border-hiplink-border dark:border-dark-border
            transform transition-transform duration-200 ease-in-out flex flex-col
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          `}
        >
          {/* Sidebar header */}
          <div className="p-4 border-b border-hiplink-border dark:border-dark-border flex items-center justify-between">
            <h2 className="font-semibold text-hiplink-dark dark:text-dark-text">Chat History</h2>
            <button
              onClick={startNewChat}
              className="flex items-center gap-1.5 text-sm font-medium text-hiplink-blue dark:text-sky-400 hover:text-hiplink-blue-dark dark:hover:text-sky-300 transition-colors"
              title="New chat"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Chat
            </button>
          </div>

          {/* Sessions list */}
          <div className="flex-1 overflow-y-auto">
            {sessionsLoading ? (
              <div className="flex items-center justify-center py-8">
                <span className="text-sm text-hiplink-secondary dark:text-dark-text-dim">Loading...</span>
              </div>
            ) : sessions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
                <p className="text-sm text-hiplink-secondary dark:text-dark-text-dim">No conversations yet.</p>
                <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mt-1">Start a new chat to see it here.</p>
              </div>
            ) : (
              <ul className="divide-y divide-hiplink-border dark:divide-dark-border">
                {sessions.map(session => (
                  <li key={session.id}>
                    <button
                      onClick={() => selectSession(session.id)}
                      className={`w-full text-left px-4 py-3 hover:bg-hiplink-background dark:hover:bg-dark-elevated transition-colors group ${
                        activeSessionId === session.id ? 'bg-hiplink-background dark:bg-dark-elevated' : ''
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className={`text-sm font-medium truncate ${
                            activeSessionId === session.id
                              ? 'text-hiplink-blue dark:text-sky-400'
                              : 'text-hiplink-dark dark:text-dark-text'
                          }`}>
                            {session.title}
                          </p>
                          <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mt-0.5">
                            {formatRelativeTime(session.updated_at)}
                          </p>
                        </div>
                        {/* Session actions */}
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={(e) => { e.stopPropagation(); archiveSession(session.id) }}
                            className="p-1 text-hiplink-secondary dark:text-dark-text-dim hover:text-hiplink-blue dark:hover:text-sky-400 rounded"
                            title="Archive"
                          >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                            </svg>
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); deleteSession(session.id) }}
                            className="p-1 text-hiplink-secondary dark:text-dark-text-dim hover:text-hiplink-error dark:hover:text-red-400 rounded"
                            title="Delete"
                          >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>

        {/* Mobile overlay */}
        {sidebarOpen && (
          <div
            className="lg:hidden fixed inset-0 bg-black/30 z-30"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main chat area */}
        <main className="flex-1 flex flex-col min-w-0">
          {/* Mode Selector */}
          <div className="p-4 bg-white dark:bg-dark-card border-b border-hiplink-border dark:border-dark-border">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 max-w-4xl mx-auto">
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

          {/* Chat container */}
          <div className="flex-1 flex flex-col overflow-hidden">
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
                      <div id="sources-section" className="mt-4 pt-4 border-t border-hiplink-border dark:border-dark-border">
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

                    {/* Suggested follow-ups (Phase 20B) */}
                    {msg.role === 'assistant' && msg.suggested_followups && msg.suggested_followups.length > 0 && (
                      <SuggestedFollowups
                        suggestions={msg.suggested_followups}
                        onSelect={handleSuggestionSelect}
                      />
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
              <form onSubmit={handleSubmit} className="flex gap-3 max-w-4xl mx-auto">
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
        </main>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function safeParseJson(json: string): unknown {
  try { return JSON.parse(json) } catch { return null }
}

function safeParseGroupedSources(json: string): GroupedSource[] {
  try { return JSON.parse(json) } catch { return [] }
}

function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString()
}

// ---------------------------------------------------------------------------
// Welcome state component
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Source card component
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Debug info panel
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Feedback buttons component
// ---------------------------------------------------------------------------

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
      const res = await authFetch(`/api/chat/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ observation_id: observationId, rating }),
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
      const res = await authFetch(`/api/chat/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          observation_id: observationId,
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

// ---------------------------------------------------------------------------
// Suggested follow-ups component (Phase 20B)
// ---------------------------------------------------------------------------

function SuggestedFollowups({ suggestions, onSelect }: { suggestions: Suggestion[]; onSelect: (suggestion: Suggestion) => void }) {
  // Phase 20C: Defensive handling - normalize and filter to safe items only
  if (!suggestions || !Array.isArray(suggestions)) {
    return null
  }

  // Filter to only valid suggestions with required fields
  const visibleSuggestions = suggestions.filter((s): s is Suggestion => {
    return (
      s !== null &&
      typeof s === 'object' &&
      typeof s.label === 'string' &&
      s.label.length > 0 &&
      typeof s.type === 'string' &&
      ['question', 'frontend_action', 'contextual_action'].includes(s.type)
    )
  })

  if (visibleSuggestions.length === 0) {
    return null
  }

  return (
    <div className="mt-4 pt-3 border-t border-hiplink-border dark:border-dark-border">
      <p className="text-xs text-hiplink-secondary dark:text-dark-text-dim mb-2">Suggested follow-ups:</p>
      <div className="flex flex-wrap gap-2">
        {visibleSuggestions.map((suggestion, i) => (
          <button
            key={i}
            onClick={() => onSelect(suggestion)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors text-left ${
              suggestion.type === 'frontend_action'
                ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800 text-green-700 dark:text-green-400 hover:bg-green-100 dark:hover:bg-green-900/50'
                : 'bg-hiplink-background dark:bg-dark-elevated border-hiplink-border dark:border-dark-border text-hiplink-blue dark:text-sky-400 hover:bg-blue-50 dark:hover:bg-sky-900/30 hover:border-hiplink-blue dark:hover:border-sky-400'
            }`}
            title={suggestion.type === 'frontend_action' ? 'Takes effect immediately' : 'Sends to AI'}
          >
            {suggestion.label}
          </button>
        ))}
      </div>
    </div>
  )
}