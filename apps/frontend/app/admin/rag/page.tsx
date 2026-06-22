'use client'

import { useState, useEffect } from 'react'
import { useAuthFetch } from '@/hooks/useApi'

interface EmbeddingStatus {
  active_provider: string
  active_model: string
  active_dimension: number
  normalize: boolean
  batch_size: number
  device: string
  collection_name: string
  collection_dimension: number | string
  reindex_required: boolean | string
  future_providers: Record<string, string>
}

interface ProviderStatus {
  available_providers: Record<string, string[]>
  active_providers: Record<string, string>
  langchain_available: boolean
  langchain_enabled: boolean
  provider_switching_ready: boolean
  unsupported_providers_disabled: boolean
  embedding_status: EmbeddingStatus
}

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
  provider_status: ProviderStatus
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
      <div className="flex items-center justify-center py-12">
        <div className="animate-pulse text-hiplink-secondary dark:text-dark-text-muted">Loading...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card dark:bg-dark-card p-6 text-center">
        <p className="text-hiplink-error dark:text-red-400">Failed to load RAG configuration: {error}</p>
      </div>
    )
  }

  if (!config) return null

  return (
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
            <ConfigRow label="Provider" value={config.provider_status.embedding_status.active_provider} highlight="blue" />
            <ConfigRow label="Model" value={config.provider_status.embedding_status.active_model} highlight="neutral" />
            <ConfigRow label="Dimension" value={config.provider_status.embedding_status.active_dimension} highlight="neutral" />
            <ConfigRow label="Normalize" value={config.provider_status.embedding_status.normalize} highlight="neutral" />
            <ConfigRow label="Batch Size" value={config.provider_status.embedding_status.batch_size} highlight="neutral" />
            <ConfigRow label="Device" value={config.provider_status.embedding_status.device} highlight="neutral" />
          </div>
          <div className="mt-3 pt-3 border-t border-hiplink-border dark:border-dark-border">
            <p className="text-xs font-medium text-hiplink-dark dark:text-dark-text mb-2">Qdrant Collection</p>
            <div className="space-y-1">
              <ConfigRow label="Collection" value={config.provider_status.embedding_status.collection_name} highlight="neutral" />
              <ConfigRow
                label="Collection Dimension"
                value={config.provider_status.embedding_status.collection_dimension}
                highlight="neutral"
              />
              <ConfigRow
                label="Reindex Required"
                value={
                  config.provider_status.embedding_status.reindex_required === true
                    ? 'Yes — reindex needed'
                    : config.provider_status.embedding_status.reindex_required === false
                    ? 'No'
                    : 'Unknown'
                }
                highlight={
                  config.provider_status.embedding_status.reindex_required === true
                    ? 'error'
                    : config.provider_status.embedding_status.reindex_required === false
                    ? 'success'
                    : 'warning'
                }
              />
            </div>
          </div>
          {/* Reindex warning */}
          {(config.provider_status.embedding_status.reindex_required === true ||
            config.provider_status.embedding_status.reindex_required === 'unknown') && (
            <div className="mt-3 p-3 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800">
              <p className="text-xs text-yellow-700 dark:text-yellow-400">
                <strong>Warning:</strong> Changing the embedding model or dimension requires a full document reindex.
                All existing vectors will be deleted and recreated with the new model.
                Run <code className="bg-yellow-100 dark:bg-yellow-900/40 px-1 rounded">python scripts/reindex_embeddings.py</code> for details.
              </p>
            </div>
          )}
          {/* Future embedding providers */}
          <div className="mt-3 pt-3 border-t border-hiplink-border dark:border-dark-border">
            <p className="text-xs font-medium text-hiplink-dark dark:text-dark-text mb-2">Available Embedding Providers</p>
            <div className="space-y-1">
              <ConfigRow label="local (MiniLM)" value="Active" highlight="success" />
              {Object.entries(config.provider_status.embedding_status.future_providers).map(([provider, status]) => (
                <ConfigRow key={provider} label={provider.toUpperCase()} value={status === 'planned' ? 'Planned / Not enabled' : status} highlight="neutral" />
              ))}
            </div>
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

        {/* Provider Interface Foundation */}
        <div className="card dark:bg-dark-card p-4">
          <h2 className="text-lg font-semibold text-hiplink-dark dark:text-dark-text mb-3">Provider Interface Foundation</h2>
          <div className="space-y-1">
            <ConfigRow
              label="Provider Switching"
              value={config.provider_status.provider_switching_ready ? 'Ready (Foundation Only)' : 'Not Ready'}
              highlight={config.provider_status.provider_switching_ready ? 'success' : 'error'}
            />
            <ConfigRow
              label="LangChain Available"
              value={config.provider_status.langchain_available ? 'Yes' : 'No'}
              highlight={config.provider_status.langchain_available ? 'warning' : 'neutral'}
            />
            <ConfigRow
              label="Unsupported Providers"
              value={config.provider_status.unsupported_providers_disabled ? 'Disabled' : 'Allowed'}
              highlight={config.provider_status.unsupported_providers_disabled ? 'success' : 'error'}
            />
          </div>
          <div className="mt-3 pt-3 border-t border-hiplink-border dark:border-dark-border">
            <p className="text-xs font-medium text-hiplink-dark dark:text-dark-text mb-2">Available Providers</p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {Object.entries(config.provider_status.available_providers).map(([key, values]) => (
                <div key={key} className="flex items-center gap-1">
                  <span className="text-hiplink-secondary dark:text-dark-text-muted capitalize">{key.replace('_', ' ')}:</span>
                  <span className="font-medium text-hiplink-dark dark:text-dark-text">{values.join(', ') || '—'}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Note */}
        <p className="text-xs text-hiplink-secondary dark:text-dark-text-muted text-center">
          This page is read-only. RAG settings are configured via environment variables.
        </p>
      </div>
    )
  }