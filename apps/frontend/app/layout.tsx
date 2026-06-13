import type { Metadata } from 'next'
import { AuthProvider } from '@/contexts/AuthContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import './globals.css'

export const metadata: Metadata = {
  title: 'HipLink AI Assistant',
  description: 'Enterprise Knowledge Assistant',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-hiplink-background text-hiplink-dark dark:bg-slate-900 dark:text-slate-100 transition-colors">
        <ThemeProvider>
          <AuthProvider>
            {children}
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}