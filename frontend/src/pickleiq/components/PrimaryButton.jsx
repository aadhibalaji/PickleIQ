import React from 'react'

// Glass rendering (backdrop blur, SVG distortion bonus, hover sheen) lives
// entirely in pickleiq.css as ::before/::after on .pickleiq-primary-button —
// same mechanism shared by every other consumer-side button (secondary
// buttons, segmented toggles, nav pills), so this stays a plain button with
// no extra DOM layers to keep in sync.
export default function PrimaryButton({ children, className = '', ...props }) {
  return (
    <button className={`pickleiq-primary-button ${className}`.trim()} {...props}>
      {children}
    </button>
  )
}
