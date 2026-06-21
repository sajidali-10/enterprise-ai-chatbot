'use client'

import { useState, useEffect } from 'react'
import { useAuthFetch } from '@/hooks/useApi'
import AdminLayout from '@/app/admin/layout'

interface RAGConfig {
  rag_pipeline_provider: string
  document_loader_provider: string
  text_splitter_provider: string
  retriever_provider: string
  reranker_provider: string
  embedding_provider: string
  embedding_model: string
  embedding_dimension: number
  vector_store_provider: string
  top_k: number
  score_threshold: number
  chunk_size: number
  chunk_overlap: number
  context_max_chunks: number
  context_max_characters: number
  conversation_context_enabled: boolean
  conversation_context_max_messages: number
  conversation_context_max_characters: number
  langchain_enabled: boolean
  reranker_enabled: boolean
}

interface ConfigRowProps {
  label: string
  value: string | number | boolean
  highlight?: 'success' | 'error' | 'warning' | 'blue' | 'neutral'
}

function ConfigRow({ label, value, highlight = 'neutral' }: ConfigRowProps) {
  const colors: Record<string, string> = {
    success: 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400',
    error: 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400',
    warning: 'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400',
    blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400',
    neutral: 'bg-gray-50 dark:bg-dark-elevated text-hiplink-secondary dark:text-dark-text-muted',
  }

  const displayValue = typeof value === 'boolean' ? (value ? 'Enabled' : 'Disabled') : String(value)

  return (
    <div className="flex items-center justify-between py-2 px-3 rounded-lg">
      <span className="text-sm text-hiplink-secondary dark:text-dark-text-muted">{label}</span>
      <span className={`px-2 py-0.5 rounded text-sm font-medium ${colors[highlight]}`}>
        {displayValue}
      </span>
    </div>
  )
}

function SectionHeader({ title }: { title: string }) {
  return (
    <h3 className="text-base font-semibold text-hiplink-dark dark:text-dark-text mb-2 mt-4">
      {title}
    </h3>
  )
}

export default function RAGConfigPage() {
  const [config, setConfig] = useState<RAGConfig | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const authFetch = useAuthFetch()

  useEffect(() => {
    authFetch('/api/admin/rag/config')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(data => {
        setConfig(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message || 'Failed to load RAG configuration')
        setLoading(false)
      })
  }, [authFetch])

  if (loading) {
    return (
      <AdminLayout>
        <div className="flex items-center justify-center py-12">
          <div className="animate-pulse text-hiplink-secondary dark:text-dark-text-muted">Loading...</div>
        </div>
      </AdminLayout>
    )
  }

  if (error) {
    return (
      <AdminLayout>
        <div className="card dark:bg-dark-card p-6 text-center">
          <p className="text-hiplink-error dark:text-red-400">Failed to load RAG configuration: {error}</p>
        </div>
      </AdminLayout>
    )
  }

  if (!config) return null

  return (
    <AdminLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h1 className="text-2xl font-bold text-hiplink-dark dark:text-dark-text">RAG Configuration</h1>
          <p className="text-sm text-hiplink-secondary dark:text-dark-text-muted mt-1">
            Current RAG pipeline settings and providers
          </p>
        </div>

        {/* Pipeline Providers */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Pipeline Providers</h2>
          <div className="space-y-1">
            <ConfigRow label="RAG Pipeline" value={config.rag_pipeline_provider} highlight="blue" />
            <ConfigRow label="Document Loader" value={config.document_loader_provider} highlight="blue" />
            <ConfigRow label="Text Splitter" value={config.text_splitter_provider} highlight="blue" />
            <ConfigRow label="Retriever" value={config.retriever_provider} highlight="blue" />
            <ConfigRow label="Reranker" value={config.reranker_provider} highlight={config.reranker_enabled ? 'warning' : 'neutral'} />
          </div>
        </div>

        {/* Embedding Settings */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Embedding</h2>
          <div className="space-y-1">
            <ConfigRow label="Provider" value={config.embedding_provider} highlight="blue" />
            <ConfigRow label="Model" value={config.embedding_model} highlight="neutral" />
            <ConfigRow label="Dimension" value={config.embedding_dimension} highlight="neutral" />
          </div>
        </div>

        {/* Vector Store */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Vector Store</h2>
          <div className="space-y-1">
            <ConfigRow label="Provider" value={config.vector_store_provider} highlight="blue" />
          </div>
        </div>

        {/* Retrieval Behavior */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Retrieval Behavior</h2>
          <div className="space-y-1">
            <ConfigRow label="Top K" value={config.top_k} highlight="neutral" />
            <ConfigRow label="Score Threshold" value={config.score_threshold} highlight="neutral" />
          </div>
        </div>

        {/* Chunking */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Chunking</h2>
          <div className="space-y-1">
            <ConfigRow label="Chunk Size" value={config.chunk_size} highlight="neutral" />
            <ConfigRow label="Chunk Overlap" value={config.chunk_overlap} highlight="neutral" />
          </div>
        </div>

        {/* Context */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Context</h2>
          <div className="space-y-1">
            <ConfigRow label="Max Chunks" value={config.context_max_chunks} highlight="neutral" />
            <ConfigRow label="Max Characters" value={config.context_max_characters} highlight="neutral" />
          </div>
        </div>

        {/* Conversation Context */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Conversation Context</h2>
          <div className="space-y-1">
            <ConfigRow
              label="Enabled"
              value={config.conversation_context_enabled}
              highlight={config.conversation_context_enabled ? 'success' : 'neutral'}
            />
            <ConfigRow label="Max Messages" value={config.conversation_context_max_messages} highlight="neutral" />
            <ConfigRow label="Max Characters" value={config.conversation_context_max_characters} highlight="neutral" />
          </div>
        </div>

        {/* Feature Flags */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Feature Flags</h2>
          <div className="space-y-1">
            <ConfigRow
              label="LangChain"
              value={config.langchain_enabled}
              highlight={config.langchain_enabled ? 'warning' : 'neutral'}
            />
            <ConfigRow
              label="Reranker"
              value={config.reranker_enabled}
              highlight={config.reranker_enabled ? 'warning' : 'neutral'}
            />
          </div>
        </div>

        {/* Note */}
        <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted text-center">
          This page is read-only. RAG settings are configured via environment variables.
        </p>
      </div>
    </AdminLayout>
  )
}