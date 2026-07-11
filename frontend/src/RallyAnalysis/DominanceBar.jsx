import React from 'react'

// Two-sided comparison bar: a blue(left)-to-orange(right) gradient track with
// a marker showing which side dominates a metric.
// Props: label, leftLabel, rightLabel, leftValue, rightValue (0..1 each) —
// or pass `score` directly (-1..1, negative = left dominates).
export default function DominanceBar({ label, leftLabel, rightLabel, leftValue, rightValue, score }) {
  const s = score != null
    ? score
    : clamp(safeDiv(rightValue - leftValue, rightValue + leftValue), -1, 1)
  const pct = ((s + 1) / 2) * 100

  return (
    <div className="dominance-bar">
      <div className="row spread">
        <b>{label}</b>
        <span className="muted">
          {leftValue != null ? `${Math.round(leftValue * 100)}%` : ''}
          {leftValue != null && rightValue != null ? ' vs ' : ''}
          {rightValue != null ? `${Math.round(rightValue * 100)}%` : ''}
        </span>
      </div>
      <div className="dominance-bar-track">
        <div className="dominance-bar-marker" style={{ left: `${pct}%` }} />
      </div>
      <div className="row spread muted" style={{ fontSize: 12 }}>
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  )
}

function safeDiv(n, d) { return d ? n / d : 0 }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
