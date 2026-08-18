'use client'

import { useRef, ChangeEvent } from 'react'

/**
 * Supported image types for direct chat attachment (Phase 34A.2).
 * Matches the backend ALLOWED_EXTENSIONS for direct-image MIME types.
 */
export const ACCEPTED_IMAGE_TYPES = [
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
] as const

export interface ChatAttachmentButtonProps {
  /** Called when a file is selected */
  onFileSelect: (file: File) => void
  /** Disabled while upload/chat is in progress */
  disabled?: boolean
}

/**
 * ChatAttachmentButton
 *
 * A paperclip icon button that opens a file picker for supported
 * image types (PNG, JPG/JPEG, WEBP).
 *
 * Renders as a small, accessible icon button beside the chat input.
 */
export function ChatAttachmentButton({
  onFileSelect,
  disabled = false,
}: ChatAttachmentButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleClick = () => {
    inputRef.current?.click()
  }

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      const file = files[0]
      onFileSelect(file)
    }
    // Reset so the same file can be re-selected
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_IMAGE_TYPES.join(',')}
        onChange={handleChange}
        className="hidden"
        id="chat-image-attachment"
        aria-label="Attach an image for OCR analysis"
      />
      <button
        type="button"
        onClick={handleClick}
        disabled={disabled}
        className={`
          p-2.5 rounded-lg transition-colors flex-shrink-0
          ${
            disabled
              ? 'text-gray-300 dark:text-gray-600 cursor-not-allowed'
              : 'text-hiplink-secondary dark:text-dark-text-muted hover:text-hiplink-blue dark:hover:text-sky-400 hover:bg-blue-50 dark:hover:bg-sky-900/30'
          }
        `}
        title="Attach screenshot or image"
        aria-label="Attach an image for analysis"
      >
        {/* Paperclip icon */}
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
        </svg>
      </button>
    </>
  )
}