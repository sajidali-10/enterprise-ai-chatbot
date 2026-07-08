'use client'

import { useEffect, useState } from 'react'
import { useTheme } from '@/contexts/ThemeContext'

export type HipLinkLogoVariant = 'auto' | 'light-bg' | 'dark-bg'
export type HipLinkLogoSize = 'sm' | 'md' | 'lg' | 'hero'

export interface HipLinkLogoProps {
  /**
   * "auto"    -> pick variant based on the current theme (dark/light).
   * "light-bg" -> logo sits on a light background, so use the dark-text variant.
   * "dark-bg"  -> logo sits on a dark background, so use the light-text variant.
   *
   * When the background is known and stable (login page, hero card), pass an
   * explicit variant to avoid theme-detection flicker.
   */
  variant?: HipLinkLogoVariant

  /**
   * Convenience size presets. Width/height are derived from the source asset's
   * aspect ratio (~1.818:1) so the wordmark is never cropped or stretched.
   *
   *   sm   ~ 120 x 66   — header
   *   md   ~ 150 x 83   — header (large), sidebar
   *   lg   ~ 180 x 99   — login page
   *   hero ~ 220 x 121  — hero / dashboard welcome
   *
   * Pass explicit `width` / `height` to override.
   */
  size?: HipLinkLogoSize

  /** Rendered pixel width. Overrides the `size` preset when provided. */
  width?: number

  /** Rendered pixel height. Overrides the `size` preset when provided. */
  height?: number

  /** Extra Tailwind/CSS classes appended to the wrapper <span>. */
  className?: string

  /**
   * Set true for above-the-fold logos (e.g. login page, hero). Disables
   * lazy loading so the wordmark renders immediately.
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

const LOGO_LIGHT_BG = '/logos/hiplink-logo-light-bg.svg'
const LOGO_DARK_BG = '/logos/hiplink-logo-dark-bg.svg'

// Source asset viewBox (800 x 440). The aspect ratio (~1.818:1) is used to
// derive size-preset heights so the wordmark is never cropped or stretched.
// Rendered width/height are independent of these intrinsic values — the SVG
// scales cleanly to any size because it is vector.
const LOGO_INTRINSIC_WIDTH = 800
const LOGO_INTRINSIC_HEIGHT = 440
const LOGO_ASPECT = LOGO_INTRINSIC_WIDTH / LOGO_INTRINSIC_HEIGHT // ~1.818

// Size presets — width drives the layout footprint; height is computed to
// preserve the source aspect ratio (so the logo is never cropped, stretched,
// or distorted). Tuned for the most common placements.
const SIZE_PRESETS: Record<HipLinkLogoSize, { width: number; height: number }> = {
  sm: { width: 120, height: Math.round(120 / LOGO_ASPECT) }, // 66
  md: { width: 150, height: Math.round(150 / LOGO_ASPECT) }, // 83
  lg: { width: 180, height: Math.round(180 / LOGO_ASPECT) }, // 99
  hero: { width: 220, height: Math.round(220 / LOGO_ASPECT) }, // 121
}

/**
 * Theme-aware HipLink wordmark.
 *
 * - Server and first client render always use the light-bg variant, matching
 *   the default theme state in `ThemeContext`. After hydration we re-resolve
 *   based on `theme` from the existing ThemeProvider, so there is no
 *   hydration mismatch.
 * - When an explicit `variant` is passed, the component is fully deterministic
 *   and does not depend on theme detection.
 * - The blue swoosh is baked into both SVG variants — it stays blue regardless
 *   of background.
 * - `width` / `height` follow the source asset's 1.818:1 aspect ratio so the
 *   wordmark is never cropped or stretched. Callers should pick a size preset
 *   (`sm` / `md` / `lg` / `hero`) rather than guessing square dimensions.
 * - Uses a plain `<img>` for SVG because vector assets do not benefit from
 *   Next.js image optimisation, and the wrapper has no aspect-ratio container
 *   to fill — width/height are explicit on every render.
 */
export default function HipLinkLogo({
  variant = 'auto',
  size = 'md',
  width,
  height,
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

  const preset = SIZE_PRESETS[size]
  const renderedWidth = width ?? preset.width
  const renderedHeight = height ?? preset.height

  const src = resolved === 'dark-bg' ? LOGO_DARK_BG : LOGO_LIGHT_BG
  const altText = decorative ? '' : (alt ?? 'HipLink')
  const ariaHidden = decorative ? true : undefined

  return (
    <span
      className={['inline-flex', 'shrink-0', 'items-center', className]
        .filter(Boolean)
        .join(' ')}
      data-logo-variant={resolved}
      data-logo-size={size}
      style={{ lineHeight: 0 }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={altText}
        width={renderedWidth}
        height={renderedHeight}
        // width/height set both the rendered size and the browser's aspect
        // ratio reservation, so there is no layout shift on load. Vector SVG
        // is rendered crisp at any pixel size; `height: auto` is not needed
        // because we always provide an explicit height. max-w-full keeps the
        // logo inside narrow viewports when explicit overrides are used.
        loading={priority ? 'eager' : 'lazy'}
        decoding={priority ? 'sync' : 'async'}
        fetchPriority={priority ? 'high' : 'auto'}
        aria-hidden={ariaHidden}
        className="block max-w-full select-none"
        draggable={false}
        style={{ height: 'auto' }}
      />
    </span>
  )
}
