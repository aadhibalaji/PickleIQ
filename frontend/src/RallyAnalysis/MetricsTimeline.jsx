import React, { useState } from 'react'

// Visualizes coaching.debug_timeline's output: the LITERAL frame-by-frame
// signal every spike is detected from. Detection is INDEPENDENT per metric
// (2026-08-05 revision) — a frame is hot when ANY metric individually
// clears ITS OWN threshold, not when a combined sum does — so the primary
// line plotted here is max_excess (the worst |deviation|/own-threshold
// fraction across metrics that frame): 1.0 is exactly "some metric just
// crossed its own bar," not an arbitrary shared number. Detected spikes are
// shaded — answers "why did (or didn't) this moment get coached" at the
// resolution it's actually computed at. Debugging/calibration view, not
// end-user content.

const CANDIDATE_COLOR = {
  flagged: '#ef5350',   // real flag, sent to the LLM
  fill: '#9e9e9e',      // below-threshold filler (see FILL_TO_LIMIT — inert
                          // under the frame-spike model, kept for schema parity)
}
const UNSELECTED_COLOR = '#7c8695'
const EXCESS_LINE_COLOR = '#58a6ff'   // the raw continuous signal — NOT a selection
                                        // indicator, drawn the same regardless of
                                        // whether anything nearby was picked
const SELECTED_OPACITY = 0.55          // sent to the LLM — should visually dominate
const SELECTED_ACTIVE_OPACITY = 0.8    // the one currently clicked open below
const UNSELECTED_TICK_OPACITY = 0.5    // detected but not selected — kept as a small
                                        // bottom-axis tick, not a full band, so it
                                        // reads as "background" rather than competing
                                        // with the selected spikes for attention

// Ticks (unselected spikes) closer together than TICK_DEDUP_GAP_S get
// deduped down to one clickable target per cluster — with 255 spikes on a
// real session, many ticks land close enough in time that clicking through
// each one individually is redundant noise, not useful detail. Greedy, same
// shape as the backend's own MIN_TIP_GAP_S suppression: sort by time, keep a
// tick selectable only if it's more than the gap past the last SELECTABLE
// one; everything else in that cluster still renders (so the density is
// still visible) but doesn't respond to clicks.
const TICK_DEDUP_GAP_S = 5.0

function markSelectableTicks(ticks) {
  const sorted = [...ticks].sort((a, b) => (a.timestamp_s ?? 0) - (b.timestamp_s ?? 0))
  let lastSelectableT = -Infinity
  const selectableIds = new Set()
  for (const s of sorted) {
    const t = s.timestamp_s ?? 0
    if (t - lastSelectableT >= TICK_DEDUP_GAP_S) {
      selectableIds.add(s.contact_id)
      lastSelectableT = t
    }
  }
  return selectableIds
}

function scaleX(minT, maxT, width, pad) {
  const span = maxT - minT || 1
  return (t) => pad + ((t - minT) / span) * (width - 2 * pad)
}

function scaleY(maxV, height, pad) {
  const span = maxV || 1
  return (v) => height - pad - (v / span) * (height - 2 * pad)
}

// One player's continuous max_excess-vs-1.0 line, with detected-spike
// ranges shaded and clickable. 1.0 is the flag line for every player/focus
// (each metric's own threshold is baked into max_excess already), so unlike
// the retired sum-based view this axis means the same thing everywhere.
function FrameExcessChart({ series, spikes, onSelect, selectedId }) {
  const W = 900, H = 160, PAD = 26
  if (!series.length) return <div className="clip-list-empty">No frame data.</div>
  const minT = series[0].timestamp_s, maxT = series[series.length - 1].timestamp_s
  const maxV = Math.max(1.5, ...series.map((fs) => fs.max_excess), 0.5) * 1.1
  const x = scaleX(minT, maxT, W, PAD)
  const y = scaleY(maxV, H, PAD)
  const poly = series.map((fs) => `${x(fs.timestamp_s)},${y(fs.max_excess)}`).join(' ')

  const unselectedTicks = spikes.filter((s) => !s.selected)
  const selectableTickIds = markSelectableTicks(unselectedTicks)

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="metrics-timeline-chart">
      {/* Unselected spikes first, as small bottom-axis ticks — detected but
          didn't make the LLM shortlist. Deliberately low-weight (not a full
          band) so they read as background, not competing with selected ones.
          Ticks within TICK_DEDUP_GAP_S of another (already-clickable) tick
          render but aren't clickable themselves — every tick still shows the
          density, but a cluster only needs one click target, not one per tick. */}
      {unselectedTicks.map((s) => {
        const bx = x(s.start_s ?? s.timestamp_s)
        const bw = Math.max(2, x(s.end_s ?? s.timestamp_s) - bx)
        const selectable = selectableTickIds.has(s.contact_id)
        return (
          <rect key={s.contact_id} x={bx} y={H - PAD - 3} width={bw} height={4}
                fill={UNSELECTED_COLOR} opacity={UNSELECTED_TICK_OPACITY} rx="1"
                onClick={selectable ? () => onSelect(s.contact_id) : undefined}
                style={{ cursor: selectable ? 'pointer' : 'default' }}>
            <title>{`${s.category} · t=${(s.timestamp_s ?? 0).toFixed(1)}s · detected, not sent to the LLM`
              + (selectable ? '' : ' (too close to another tick to select separately)')}</title>
          </rect>
        )
      })}
      {/* Selected spikes: full-height, bright, on top — these are the ≤16
          that actually reached the LLM. */}
      {spikes.filter((s) => s.selected).map((s) => (
        <rect key={s.contact_id}
              x={x(s.start_s ?? s.timestamp_s)} y={PAD - 4}
              width={Math.max(2, x(s.end_s ?? s.timestamp_s) - x(s.start_s ?? s.timestamp_s))}
              height={H - PAD * 2 + 8}
              fill={CANDIDATE_COLOR[s.candidate_type] || UNSELECTED_COLOR}
              stroke={CANDIDATE_COLOR[s.candidate_type] || UNSELECTED_COLOR}
              strokeWidth={s.contact_id === selectedId ? 1.5 : 0}
              opacity={s.contact_id === selectedId ? SELECTED_ACTIVE_OPACITY : SELECTED_OPACITY}
              onClick={() => onSelect(s.contact_id)} style={{ cursor: 'pointer' }}>
          <title>{`${s.category} · t=${(s.timestamp_s ?? 0).toFixed(1)}s · ${s.deviation_score?.toFixed?.(2)} on ${s.contact_id.split('_').slice(-2, -1)[0]} · sent to the LLM`}</title>
        </rect>
      ))}
      <line x1={PAD} x2={W - PAD} y1={y(1.0)} y2={y(1.0)}
            stroke="#ef5350" strokeDasharray="4,3" strokeWidth="1" opacity="0.7" />
      <text x={W - PAD} y={y(1.0) - 4} textAnchor="end" fontSize="10" fill="#ef5350" opacity="0.85">
        some metric's own threshold (1.0)
      </text>
      <line x1={PAD} x2={W - PAD} y1={y(0)} y2={y(0)} stroke="#3a4150" strokeWidth="1" />
      <polyline points={poly} fill="none" stroke={EXCESS_LINE_COLOR} strokeWidth="1.25" />
    </svg>
  )
}

// One mini-sparkline per metric field: that metric's RAW per-frame value
// over time (not deviation) — "track the metrics over time" at the
// individual-metric level, with the acceptable band shaded and this
// metric's OWN detection line marked (dashed) — the exact raw value at
// which a reading starts a spike for THIS metric, useful for judging
// whether its threshold is set too tight/loose against real readings.
function MetricMiniChart({ field, series, band, thresholdBounds }) {
  const points = series
    .map((fs) => ({ t: fs.timestamp_s, v: fs.values?.[field] }))
    .filter((p) => p.v !== null && p.v !== undefined)
  if (points.length < 2) return null
  const W = 220, H = 56, PAD = 6
  const values = points.map((p) => p.v)
  const [lo, hi] = band || [Math.min(...values), Math.max(...values)]
  const [tLo, tHi] = thresholdBounds || [lo, hi]
  const minV = Math.min(lo, tLo, ...values), maxV = Math.max(hi, tHi, ...values)
  const x = scaleX(points[0].t, points[points.length - 1].t, W, PAD)
  const yScale = (v) => H - PAD - ((v - minV) / ((maxV - minV) || 1)) * (H - 2 * PAD)
  const poly = points.map((p) => `${x(p.t)},${yScale(p.v)}`).join(' ')

  return (
    <div className="metrics-mini-chart">
      <div className="muted" style={{ fontSize: 11 }}>{field.replace(/_/g, ' ')}</div>
      <svg width={W} height={H}>
        {band && (
          <rect x={PAD} y={yScale(hi)} width={W - 2 * PAD} height={Math.max(0, yScale(lo) - yScale(hi))}
                fill="#3fb95022" />
        )}
        {thresholdBounds && (
          <>
            <line x1={PAD} x2={W - PAD} y1={yScale(tLo)} y2={yScale(tLo)}
                  stroke="#ef5350" strokeDasharray="2,2" strokeWidth="0.75" opacity="0.6" />
            <line x1={PAD} x2={W - PAD} y1={yScale(tHi)} y2={yScale(tHi)}
                  stroke="#ef5350" strokeDasharray="2,2" strokeWidth="0.75" opacity="0.6" />
          </>
        )}
        <polyline points={poly} fill="none" stroke={EXCESS_LINE_COLOR} strokeWidth="1" />
      </svg>
    </div>
  )
}

export default function MetricsTimeline({ debug }) {
  const [selectedId, setSelectedId] = useState(null)
  if (!debug) return null
  const shots = debug.shots || []
  const frameSeries = debug.frame_series || {}
  const selected = shots.find((s) => s.contact_id === selectedId) || null
  const players = Object.keys(frameSeries)

  return (
    <div className="metrics-timeline">
      <div className="row" style={{ gap: 16, marginBottom: 4, flexWrap: 'wrap' }}>
        <span className="metrics-legend-item"><span className="metrics-dot" style={{ background: EXCESS_LINE_COLOR }} />the line: raw signal, every frame (not a selection indicator)</span>
        <span className="metrics-legend-item"><span className="metrics-dot" style={{ background: CANDIDATE_COLOR.flagged }} />tall red band: sent to the LLM</span>
        <span className="metrics-legend-item"><span className="metrics-dot" style={{ background: UNSELECTED_COLOR }} />small gray tick: detected, NOT sent</span>
      </div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
        Detection is independent per metric — a frame is hot when ANY ONE metric individually
        crosses its own line, not when several add up. Of {shots.length} spikes detected this
        session, only {shots.filter((s) => s.selected).length} were sent to the LLM (limit {debug.max_llm_candidates} candidates,
        capped at {debug.category_cap} per category so one issue can't crowd out the rest) — the small
        gray ticks along the bottom are everything that DIDN'T make that cut. Click any band or tick for detail.
      </div>
      {players.length === 0 && <div className="clip-list-empty">No frame data for this focus.</div>}
      {players.map((label) => (
        <div key={label} style={{ marginBottom: 14 }}>
          <div className="muted" style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>{label}</div>
          <FrameExcessChart series={frameSeries[label]}
                            spikes={shots.filter((s) => s.player_label === label)}
                            onSelect={setSelectedId} selectedId={selectedId} />
        </div>
      ))}

      {selected && (
        <div className="metrics-selected-detail">
          <div className="row spread">
            <b>{selected.player_label} · {selected.category} · {(selected.timestamp_s ?? 0).toFixed(1)}s</b>
            <span className="muted">
              {selected.duration_s ? `${selected.duration_s.toFixed(1)}s span` : 'instant'}
            </span>
          </div>
          <table className="rally-stats-table" style={{ marginTop: 8 }}>
            <thead><tr><th>metric</th><th>value at peak</th><th>deviation</th></tr></thead>
            <tbody>
              {Object.entries(selected.metrics).filter(([, m]) => m.value !== null || m.deviation !== null).map(([field, m]) => (
                <tr key={field}>
                  <td>{field.replace(/_/g, ' ')}</td>
                  <td>{m.value === null ? '—' : m.value.toFixed(2)}</td>
                  <td>{m.deviation === null ? '—' : m.deviation.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="muted" style={{ fontSize: 12, margin: '16px 0 6px' }}>
        Individual metrics over time (shaded = acceptable band)
      </div>
      {players.map((label) => (
        <div key={label} style={{ marginBottom: 10 }}>
          <div className="muted" style={{ fontSize: 11, fontWeight: 600 }}>{label}</div>
          <div className="metrics-mini-grid">
            {(debug.metric_fields || []).map((f) => (
              <MetricMiniChart key={f} field={f} series={frameSeries[label]}
                               band={debug.target_bands?.[f]}
                               thresholdBounds={debug.metric_threshold_bounds?.[f]} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
