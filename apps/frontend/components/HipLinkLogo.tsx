'use client'

import Image from 'next/image'
import { useEffect, useState } from 'react'
import { useTheme } from '@/contexts/ThemeContext'

export type HipLinkLogoVariant = 'auto' | 'light-bg' | 'dark-bg'

export interface HipLinkLogoProps {
  /**
   * "auto"   -> pick variant based on the current theme (dark/light).
   * "light-bg" -> logo sits on a light background, so use the dark-text variant.
   * "dark-bg"  -> logo sits on a dark background, so use the light-text variant.
   *
   * When the background is known and stable (login page, hero card), pass an
   * explicit variant to avoid theme-detection flicker.
   */
  variant?: HipLinkLogoVariant

  /** Rendered pixel width. Defaults to 130. */
  width?: number

  /** Rendered pixel height. Defaults to 48. */
  height?: number

  /** Extra Tailwind/CSS classes appended to the wrapper <span>. */
  className?: string

  /**
   * Forwarded to next/image — set true for above-the-fold logos (e.g. login page).
   * Triggers preload and removes lazy-loading delay.
   */
  priority?: boolean

  /**
   * Alt text override. Defaults to "HipLink".
   * Pass empty string only when an adjacent accessible brand label already exists,
   * and pair it with `decorative` to suppress redundant announcements.
   */
  alt?: string

  /** Mark the logo as decorative (aria-hidden + empty alt). */
  decorative?: boolean
}

const LOGO_LIGHT_BG = '/logos/hiplink-logo-light-bg.png'
const LOGO_DARK_BG = '/logos/hiplink-logo-dark-bg.png'

/**
 * Theme-aware HipLink wordmark.
 *
 * - Server and first client render always use the light-bg variant, matching the
 *   default theme state in `ThemeContext`. After hydration we re-resolve based
 *   on `theme` from the existing ThemeProvider, so there is no hydration mismatch.
 * - When an explicit `variant` is passed, the component is fully deterministic
 *   and does not depend on theme detection.
 * - The blue swoosh is baked into both PNG variants — it stays blue regardless
 *   of background.
 */
export default function HipLinkLogo({
  variant = 'auto',
  width = 130,
  height = 48,
  className,
  priority = false,
  alt,
  decorative = false,
}: HipLinkLogoProps) {
  const { theme } = useTheme()

  // Mounted flag prevents any chance of theme flicker being visible as a layout
  // jump. We default to the light-bg variant until mounted (matches SSR output).
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    setMounted(true)
  }, [])

  const resolved: 'light-bg' | 'dark-bg' =
    variant === 'light-bg' || variant === 'dark-bg'
      ? variant
      : mounted && theme === 'dark'
        ? 'dark-bg'
        : 'light-bg'

  const src = resolved === 'dark-bg' ? LOGO_DARK_BG : LOGO_LIGHT_BG
  const altText = decorative ? '' : (alt ?? 'HipLink')
  const ariaHidden = decorative ? true : undefined

  return (
    <span
      className={['inline-flex', 'shrink-0', 'items-center', className]
        .filter(Boolean)
        .join(' ')}
      data-logo-variant={resolved}
    >
      <Image
        src={src}
        alt={altText}
        width={width}
        height={height}
        priority={priority}
        aria-hidden={ariaHidden}
        className="object-contain h-auto w-auto max-w-full"
      />
    </span>
  )
}
