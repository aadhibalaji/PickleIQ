import React from 'react'

// Percentage callout: big number + label + a player initial "thumbnail".
// Props: value (0..1), label, playerLabel
export default function StatBlock({ value, label, playerLabel }) {
  const initial = playerLabel ? playerLabel.trim()[0]?.toUpperCase() : '?'
  return (
    <div className="stat-block">
      <div className="stat-block-avatar">{initial}</div>
      <div>
        <div className="stat-block-value">{value != null ? `${Math.round(value * 100)}%` : '—'}</div>
        <div className="muted">{label}{playerLabel ? ` · ${playerLabel}` : ''}</div>
      </div>
    </div>
  )
}
