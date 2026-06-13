import type { Metadata } from 'next'
import { AuthProvider } from '@/contexts/AuthContext'
import './globals.css'

export const metadata: Metadata = {
  title: 'HipLink AI Assistant',
  description: 'Enterprise Knowledge Assistant',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-hiplink-background text-hiplink-dark">
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  )
}