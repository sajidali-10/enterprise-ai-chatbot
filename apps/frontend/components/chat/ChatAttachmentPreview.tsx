'use client'

import { useState } from 'react'
import type { AttachmentState } from '@/hooks/useChatAttachment'

export interface ChatAttachmentPreviewProps {
  /** Current attachment state from useChatAttachment */
  attachment: AttachmentState
  /** Callback to remove the attachment */
  onRemove: () => void
  /** Callback to retry upload */
  onRetry?: () => void
}

/**
 * Status labels for each attachment lifecycle phase.
 * Maps the internal status to user-facing text and styling.
 */
const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  idle: { label: '', color: '' },
  selecting: { label: 'Ready', color: 'text-gray-500 dark:text-gray-400' },
  validating: { label: 'Validating...', color: 'text-amber-600 dark:text-amber-400' },
  uploading: { label: 'Uploading...', color: 'text-blue-600 dark:text-sky-400' },
  indexing: { label: 'Indexing...', color: 'text-blue-600 dark:text-sky-400' },
  ready: { label: 'Ready', color: 'text-green-600 dark:text-green-400' },
  failed: { label: 'Failed', color: 'text-red-600 dark:text-red-400' },
}

/**
 * ChatAttachmentPreview
 *
 * Compact attachment preview strip shown above or inside the chat composer.
 *
 * ┌────────────────────────────────────────────┐
 * │ [thumbnail] filename.png            [ X ]  │
 * │ [status]                                 │
 * └────────────────────────────────────────────┘
 *
 * States:
 * - Uploading / Indexing: shows spinner + status text
 * - Ready: shows green checkmark + filename
 * - Failed: shows error + retry option
 */
export function ChatAttachmentPreview({
  attachment,
  onRemove,
  onRetry,
}: ChatAttachmentPreviewProps) {
  const [imgSrc, setImgSrc] = useState<string | null>(null)

  // Generate thumbnail from the selected file
  const file = attachment.file
  const displayName = file?.name ?? 'Attached image'

  // Load thumbnail
  if (file && !imgSrc) {
    const reader = new FileReader()
    reader.onload = (e) => setImgSrc(e.target?.result as string)
    reader.readAsDataURL(file)
  }

  const statusInfo = STATUS_LABELS[attachment.status] ?? {
    label: attachment.status,
    color: 'text-gray-500',
  }

  const isRemovable =
    attachment.status !== 'uploading' &&
    attachment.status !== 'indexing'

  return (
    <div className="flex items-center gap-3 p-2 bg-hiplink-background dark:bg-dark-elevated border border-hiplink-border dark:border-dark-border rounded-lg w-full">
      {/* Thumbnail */}
      <div className="w-10 h-10 flex-shrink-0 rounded overflow-hidden bg-gray-100 dark:bg-gray-800">
        {imgSrc ? (
          <img
            src={imgSrc}
            alt={displayName}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <svg
              className="w-5 h-5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l2.586-2.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </div>
        )}
      </div>

      {/* File info + status */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-hiplink-dark dark:text-dark-text truncate">
          {displayName}
        </p>
        <p className={`text-xs ${statusInfo.color}`}>
          {statusInfo.label}
        </p>
      </div>

      {/* Remove / X button */}
      {isRemovable && (
        <button
          type="button"
          onClick={onRemove}
          className="p-1 text-hiplink-secondary dark:text-dark-text-muted hover:text-hiplink-error dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors flex-shrink-0"
          title="Remove attachment"
          aria-label="Remove attached image"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}

      {/* Error detail */}
      {attachment.status === 'failed' && attachment.error && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-50/80 dark:bg-red-900/40 rounded-lg">
          <p className="text-xs text-hiplink-error dark:text-red-400 px-3">
            {attachment.error}
          </p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="ml-2 text-xs text-hiplink-blue dark:text-sky-400 hover:underline"
            >
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  )
}