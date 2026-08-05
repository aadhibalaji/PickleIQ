import React from 'react'

export default function SegmentedControl({ label, options, value, onChange, tone = 'neutral', disabled = false }) {
  return (
    <fieldset className={`pickleiq-segmented pickleiq-segmented--${tone}`}>
      <legend className="pickleiq-visually-hidden">{label}</legend>
      {options.map((option) => {
        const selected = option.value === value
        return (
          <button key={option.value} type="button" className={selected ? 'is-selected' : ''}
            aria-pressed={selected} disabled={disabled} onClick={() => onChange(option.value)}>
            {option.label}
          </button>
        )
      })}
    </fieldset>
  )
}
