'use client'

import { useState, useRef, DragEvent, ChangeEvent } from 'react'
import Link from 'next/link'
import { useAuthFetch } from '@/hooks/useApi'

const ALLOWED_TYPES = [
  '.pdf',
  '.txt',
  '.md',
  '.docx',
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.tif',
  '.tiff',
  '.bmp',
]
const MAX_SIZE_MB = 20
const MAX_SIZE = MAX_SIZE_MB * 1024 * 1024

export default function DocumentUploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const authFetch = useAuthFetch()

  function validateFile(file: File): string | null {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!ALLOWED_TYPES.includes(ext)) {
      return `Invalid file type. Allowed types: ${ALLOWED_TYPES.join(', ')}`
    }
    if (file.size > MAX_SIZE) {
      return `File too large. Maximum size is ${MAX_SIZE_MB}MB.`
    }
    return null
  }

  function handleFile(selectedFile: File) {
    setMessage(null)
    const validationError = validateFile(selectedFile)
    if (validationError) {
      setError(validationError)
      setFile(null)
    } else {
      setError(null)
      setFile(selectedFile)
    }
  }

  function handleDrag(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  function removeFile() {
    setFile(null)
    setError(null)
    setMessage(null)
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  async function handleUpload() {
    if (!file) return

    setUploading(true)
    setError(null)
    setMessage(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await authFetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Upload failed with status ${res.status}`)
      }

      setMessage(`Successfully uploaded: ${file.name}`)
      removeFile()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold">Upload Document</h1>
          <Link
            href="/documents"
            className="text-blue-500 hover:text-blue-600 font-medium"
          >
            View Documents
          </Link>
        </div>

        <div className="bg-white shadow-lg rounded-lg p-6">
          {/* Drop Zone */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ALLOWED_TYPES.join(',')}
              onChange={handleChange}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer flex flex-col items-center"
            >
              <svg
                className="w-12 h-12 text-gray-400 mb-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <span className="text-gray-600">
                Drag and drop a file here, or{' '}
                <span className="text-blue-500 font-medium">browse</span>
              </span>
              <span className="text-sm text-gray-400 mt-2">
Allowed types: {ALLOWED_TYPES.join(', ')} | Max size: {MAX_SIZE_MB}MB | Images are OCR&#39;d
              </span>
            </label>
          </div>

          {/* Selected File */}
          {file && (
            <div className="mt-4 flex items-center justify-between bg-gray-50 rounded-lg p-3">
              <div className="flex items-center space-x-3 overflow-hidden">
                <svg
                  className="w-8 h-8 text-gray-400 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <div className="overflow-hidden">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {file.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
              <button
                onClick={removeFile}
                className="text-gray-400 hover:text-red-500 ml-2 flex-shrink-0"
                disabled={uploading}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
              {error}
            </div>
          )}

          {/* Success Message */}
          {message && (
            <div className="mt-4 bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg text-sm">
              {message}
            </div>
          )}

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className={`mt-4 w-full py-2 px-4 rounded-lg font-medium transition-colors ${
              !file || uploading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-500 text-white hover:bg-blue-600'
            }`}
          >
            {uploading ? (
              <span className="flex items-center justify-center space-x-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                <span>Uploading...</span>
              </span>
            ) : (
              'Upload Document'
            )}
          </button>
        </div>

        <Link
          href="/"
          className="block mt-4 text-center text-gray-500 hover:text-gray-700"
        >
          Back to Home
        </Link>
      </div>
    </main>
  )
}