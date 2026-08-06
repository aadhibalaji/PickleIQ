import React from 'react'

// SVG turbulence/displacement filter that gives PrimaryButton its liquid-glass
// distortion (referenced via CSS backdrop-filter: url(#pickleiq-glass-filter)
// in pickleiq.css). Mounted once per screen, in Header — filters just need to
// exist somewhere in the DOM to be referenced by id, so this stays invisible
// (visually-hidden, not display:none, so browsers keep it resolvable).
export default function GlassFilterDefs() {
  return (
    <svg aria-hidden="true" focusable="false" className="pickleiq-visually-hidden">
      <filter id="pickleiq-glass-filter" x="0%" y="0%" width="100%" height="100%"
        colorInterpolationFilters="sRGB">
        <feTurbulence type="fractalNoise" baseFrequency="0.05 0.05" numOctaves="1" seed="1" result="turbulence" />
        <feGaussianBlur in="turbulence" stdDeviation="2" result="blurredNoise" />
        <feDisplacementMap in="SourceGraphic" in2="blurredNoise" scale="70"
          xChannelSelector="R" yChannelSelector="B" result="displaced" />
        <feGaussianBlur in="displaced" stdDeviation="4" result="finalBlur" />
        <feComposite in="finalBlur" in2="finalBlur" operator="over" />
      </filter>
    </svg>
  )
}
