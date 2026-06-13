import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        hiplink: {
          blue: '#2F80C9',
          'blue-dark': '#2171B5',
          'blue-light': '#4A9DE0',
          dark: '#111827',
          secondary: '#6B7280',
          background: '#F8FAFC',
          card: '#FFFFFF',
          border: '#E5E7EB',
          success: '#16A34A',
          warning: '#D97706',
          error: '#DC2626',
        },
      },
    },
  },
  plugins: [],
}

export default config