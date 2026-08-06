import React from 'react'

// Visualizes coaching._frame_general_context (part of the same
// debug_timeline payload): per player, per category, the metric's SESSION
// AVERAGE compared to the same absolute target used for spike detection —
// the deterministic aggregate the LLM's praise/improve picks
// (report.general_feedback) are supposed to be grounded in. Shows AT A
// GLANCE whether a player's "Focus on" pick genuinely matches their worst
// session average, or whether the general feedback text drifted from what
// the numbers actually show.

const NEGATIVE_COLOR = '#ef5350'
const POSITIVE_COLOR = '#66bb6a'
const NEUTRAL_COLOR = '#5b6472'

// Position of `value` along [lo - pad, hi + pad] as a 0-1 fraction, so the
// marker can sit outside the band when the average itself is out of range.
function fractionInRange(value, lo, hi) {
  const pad = (hi - lo || 1) * 0.4
  const min = lo - pad, max = hi + pad
  return Math.max(0, Math.min(1, (value - min) / (max - min || 1)))
}

function CategoryRow({ category, avgValue, deviation, nFrames, band, isNegativePick, isPositivePick }) {
  const color = isNegativePick ? NEGATIVE_COLOR : isPositivePick ? POSITIVE_COLOR : NEUTRAL_COLOR
  const [lo, hi] = band || [null, null]
  const hasBand = lo !== null && hi !== null
  const bandLoFrac = hasBand ? fractionInRange(lo, lo, hi) : 0
  const bandHiFrac = hasBand ? fractionInRange(hi, lo, hi) : 1
  const markerFrac = hasBand ? fractionInRange(avgValue, lo, hi) : 0.5
  return (
    <div className="general-breakdown-row">
      <div className="general-breakdown-label">
        {category.replace(/_/g, ' ')}
        {isNegativePick && <span title="strongest_negative_category — drives the 'Focus on' pick"> 🎯</span>}
        {isPositivePick && <span title="strongest_positive_category — drives the 'What's working' pick"> 👍</span>}
      </div>
      <div className="general-breakdown-track">
        {hasBand && (
          <div className="general-breakdown-band"
               style={{ left: `${bandLoFrac * 100}%`, width: `${(bandHiFrac - bandLoFrac) * 100}%` }} />
        )}
        <div className="general-breakdown-marker" style={{ left: `${markerFrac * 100}%`, background: color }}
             title={`avg ${avgValue.toFixed(2)} · target [${hasBand ? lo.toFixed(2) : '?'}, ${hasBand ? hi.toFixed(2) : '?'}]`} />
      </div>
      <div className="muted general-breakdown-count">
        avg {avgValue.toFixed(1)} · dev {deviation.toFixed(2)} · {nFrames}f
      </div>
    </div>
  )
}

export default function GeneralFeedbackBreakdown({ generalContext, targetBands }) {
  const players = Object.entries(generalContext || {})
  if (!players.length) return <div className="clip-list-empty">No general-feedback context available.</div>

  return (
    <div className="general-breakdown">
      {players.map(([label, ctx]) => {
        const categories = Object.entries(ctx.by_category || {})
        return (
          <div key={label} className="general-breakdown-player">
            <b>{label}</b>
            {categories.length === 0 ? (
              <div className="muted">No scored moments.</div>
            ) : categories.map(([cat, stats]) => (
              <CategoryRow key={cat} category={cat} avgValue={stats.avg_value}
                           deviation={stats.deviation ?? 0} nFrames={stats.n_frames}
                           band={targetBands?.[stats.metric]}
                           isNegativePick={cat === ctx.strongest_negative_category}
                           isPositivePick={cat === ctx.strongest_positive_category} />
            ))}
            {!ctx.strongest_positive_category && (
              <div className="muted" style={{ fontSize: 12 }}>No category sampled enough to praise yet.</div>
            )}
          </div>
        )
      })}
    </div>
  )
}
