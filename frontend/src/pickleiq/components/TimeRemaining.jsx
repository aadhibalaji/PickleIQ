import React from 'react'

function formatEstimate(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return 'Estimating time remaining…'
  if (seconds < 60) return 'Less than a minute remaining'
  const minutes = Math.ceil(seconds / 60)
  return `About ${minutes} minute${minutes === 1 ? '' : 's'} remaining`
}

export default function TimeRemaining({ seconds }) {
  if (!Number.isFinite(seconds) || seconds < 0) return null
  return (
    <p className="pickleiq-time-remaining">
      {formatEstimate(seconds)}
    </p>
  )
}
