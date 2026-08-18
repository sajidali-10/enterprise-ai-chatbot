'use client'

import { useState, useCallback } from 'react'
import { useAuthFetch } from '@/hooks/useApi'

/**
 * Supported image types for direct chat attachment (Phase 34A.2).
 * Matches the backend ALLOWED_EXTENSIONS for image MIME types.
 */
export const ATTACHMENT_IMAGE_TYPES = [
  'image/png',
  'image/jpeg',
  'image/jpg',
  'image/webp',
] as const

export const ATTACHMENT_IMAGE_EXTENSIONS = [
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
] as const

export const ATTACHMENT_MAX_SIZE_MB = 20
export const ATTACHMENT_MAX_SIZE = ATTACHMENT_MAX_SIZE_MB * 1024 * 1024

type AttachmentStatus =
  | 'idle'
  | 'selecting'
  | 'validating'
  | 'uploading'
  | 'indexing'
  | 'ready'
  | 'failed'
  | 'removed'

export interface AttachmentState {
  file: File | null
  status: AttachmentStatus
  error: string | null
  documentId: number | null
  imageId: number | null
}

export interface UseChatAttachmentReturn {
  /** Current attachment state */
  attachment: AttachmentState
  /** Select and validate a file (client-side checks only) */
  selectFile: (file: File) => void
  /** Upload the selected (validated) file, poll until ready, returns identifiers */
  uploadImage: (file: File) => Promise<void>
  /** Remove the attachment (clear state) */
  removeAttachment: () => void
  /** Reset to idle */
  reset: () => void
  /** True while upload/index is in progress */
  isProcessing: boolean
}

/**
 * useChatAttachment
 *
 * Manages the lifecycle of a single image attachment for the Chat page.
 * - File selection + client-side validation (type, size)
 * - Upload to existing documents API
 * - Resolve document_id + image_id from upload response
 */
export function useChatAttachment(): UseChatAttachmentReturn {
  const [attachment, setAttachment] = useState<AttachmentState>({
    file: null,
    status: 'idle',
    error: null,
    documentId: null,
    imageId: null,
  })
  const authFetch = useAuthFetch()

  const isProcessing = attachment.status === 'uploading' || attachment.status === 'indexing'

  const selectFile = useCallback((file: File) => {
    // Validate extension
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!ATTACHMENT_IMAGE_EXTENSIONS.includes(ext as typeof ATTACHMENT_IMAGE_EXTENSIONS[number])) {
      setAttachment({
        file: null,
        status: 'failed',
        error: `Unsupported image type. Supported: ${ATTACHMENT_IMAGE_EXTENSIONS.join(', ')}`,
        documentId: null,
        imageId: null,
      })
      return
    }

    // Validate MIME
    if (!file.type.startsWith('image/')) {
      setAttachment({
        file: null,
        status: 'failed',
        error: 'File is not a supported image type',
        documentId: null,
        imageId: null,
      })
      return
    }

    // Validate size
    if (file.size > ATTACHMENT_MAX_SIZE) {
      setAttachment({
        file: null,
        status: 'failed',
        error: `File too large. Maximum size is ${ATTACHMENT_MAX_SIZE_MB}MB`,
        documentId: null,
        imageId: null,
      })
      return
    }

    // Clear any previous error, show preview
    setAttachment({
      file,
      status: 'selecting',
      error: null,
      documentId: null,
      imageId: null,
    })
  }, [])

  const removeAttachment = useCallback(() => {
    setAttachment({
      file: null,
      status: 'removed',
      error: null,
      documentId: null,
      imageId: null,
    })
  }, [])

  const reset = useCallback(() => {
    setAttachment({
      file: null,
      status: 'idle',
      error: null,
      documentId: null,
      imageId: null,
    })
  }, [])

  const uploadImage = useCallback(async (file: File) => {
    // Always set status to uploading (even if already selecting)
    setAttachment({
      file,
      status: 'uploading',
      error: null,
      documentId: null,
      imageId: null,
    })

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await authFetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(data.detail || `Upload failed: ${res.status}`)
      }

      const result = await res.json()

      // The upload response now includes:
      //   id — document_id
      //   image_id — first DocumentImage id (Phase 34A.2)
      const documentId = result.id as number
      const imageId = result.image_id as number | null

      setAttachment({
        file,
        status: 'ready',
        error: null,
        documentId,
        imageId,
      })
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Upload failed'
      setAttachment({
        file,
        status: 'failed',
        error: errorMsg,
        documentId: null,
        imageId: null,
      })
    }
  }, [authFetch])

  return {
    attachment,
    selectFile,
    uploadImage,
    removeAttachment,
    reset,
    isProcessing,
  }
}