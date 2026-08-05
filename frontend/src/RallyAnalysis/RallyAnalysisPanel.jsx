import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import ShotHeatmap from './ShotHeatmap.jsx'
import CoachingSelections from './CoachingSelections.jsx'
import CoachingSetupSelector from './CoachingSetupSelector.jsx'
import { buildPlayerColorMap, playerColor, labelToPlayerId } from './playerColors.js'

const ALL_PLAYERS = 'all'

// Mirror PRE_WINDOW_S/POST_WINDOW_S in backend/app/rallyanalysis/coaching.py
// (source of truth) — the swing-analysis window around each contact's peak
// frame, used here to draw the coaching-moment block's time span.
const PRE_WINDOW_S = 0.6
const POST_WINDOW_S = 0.45

// Coaching-moment timeline lane layout: blocks that would touch/overlap in
// time get stacked into rows instead of drawn on top of each other.
// LANE_GAP_S is a small time-domain buffer (rather than a measured pixel
// gap — the track's pixel width isn't known without a DOM measurement, and
// a tiny time buffer gives the same "don't quite touch" effect since block
// width is itself proportional to time).
const LANE_GAP_S = 0.1
const MAX_LANES = 3   // beyond this, later blocks share the last lane (drawn more transparent)
const LANE_HEIGHT_PX = 16
const LANE_ROW_GAP_PX = 4

// Greedy interval-partitioning: sort by start, place each block in the
// first lane whose last block ends before this one starts (plus the
// buffer); if none fits and we're at the lane cap, pile onto the last lane
// instead of growing forever. Pure function of [{id, start, end}] ->
// {laneById: Map<id, {lane, overflow}>, laneCount}.
function assignLanes(blocks) {
  const sorted = [...blocks].sort((a, b) => a.start - b.start)
  const laneEnds = []
  const laneById = new Map()
  for (const b of sorted) {
    let lane = laneEnds.findIndex((end) => b.start >= end + LANE_GAP_S)
    let overflow = false
    if (lane === -1) {
      if (laneEnds.length < MAX_LANES) {
        lane = laneEnds.length
        laneEnds.push(b.end)
      } else {
        lane = MAX_LANES - 1
        overflow = true
        laneEnds[lane] = Math.max(laneEnds[lane], b.end)
      }
    } else {
      laneEnds[lane] = b.end
    }
    laneById.set(b.id, { lane, overflow })
  }
  return { laneById, laneCount: laneEnds.length || 1 }
}

export default function RallyAnalysisPanel({ videoId, video, onUpdateVideo, onSeek, frame }) {
  const [report, setReport] = useState(null)
  const [generating, setGenerating] = useState(false)
  const [selectedPlayer, setSelectedPlayer] = useState(ALL_PLAYERS)
  const [selectedTip, setSelectedTip] = useState(null)
  const [stats, setStats] = useState(null)
  const [statsError, setStatsError] = useState(false)
  // {contact_id: 1 | -1} — per-tip thumbs feedback, persisted server-side so
  // tip-selection changes become a measurable hit-rate across reports.
  const [tipFeedback, setTipFeedback] = useState({})

  useEffect(() => {
    api.getCoachingReport(videoId).then((r) => setReport(r.report)).catch(() => setReport(null))
    setStats(null)
    setStatsError(false)
    api.getReportStats(videoId).then((r) => setStats(r)).catch(() => setStatsError(true))
    setTipFeedback({})
    api.getTipFeedback(videoId).then((r) => setTipFeedback(r.ratings || {})).catch(() => {})
    setSelectedPlayer(ALL_PLAYERS)
    setSelectedTip(null)
  }, [videoId])

  const onRateTip = (tip, rating) => {
    // Optimistic update; re-clicking the active rating keeps it (simplest
    // contract — the backend upserts, so repeat sends are harmless).
    setTipFeedback((prev) => ({ ...prev, [tip.contact_id]: rating }))
    api.sendTipFeedback(videoId, {
      contact_id: tip.contact_id, rating,
      category: tip.category, tone: tip.tone, tip: tip.tip,
    }).catch(() => {})
  }

  const onGenerateReport = () => {
    setGenerating(true)
    api.generateCoachingReport(videoId).then((r) => {
      setReport(r)
      setSelectedPlayer(ALL_PLAYERS)
      setSelectedTip(null)
    }).catch(() => {}).finally(() => setGenerating(false))
  }

  // Coaching-only fast path for a pre-trimmed clip that IS one rally:
  // replaces this video's pass1 rally labels with a single whole-clip rally
  // and reruns Pass 2 (which chains contact detection). Destructive to
  // existing pass1 labels, hence the confirm.
  const [markingSingleRally, setMarkingSingleRally] = useState(false)
  const onProcessSingleRally = () => {
    if (!window.confirm(
      'Treat this ENTIRE clip as one rally and reprocess Pass 2?\n\n' +
      'Only use this on a pre-trimmed single-rally clip — it replaces any ' +
      'detected rally boundaries for this video.')) return
    setMarkingSingleRally(true)
    api.processSingleRally(videoId).catch(() => {}).finally(() => setMarkingSingleRally(false))
  }

  const tips = report?.tips || []
  const players = report?.players || []
  // New reports carry per-player general feedback (one praise + one improve,
  // replacing themes); legacy reports still carry themes — render whichever
  // this report has, preferring general feedback.
  const general = report?.general_feedback || []
  const themes = report?.player_themes || []
  const visibleGeneral = selectedPlayer === ALL_PLAYERS
    ? general
    : general.filter((g) => g.player_label === selectedPlayer)
  const visibleThemes = selectedPlayer === ALL_PLAYERS
    ? themes
    : themes.filter((t) => t.player_label === selectedPlayer)

  // One color map for the whole panel (filter tabs, tip labels, stats,
  // sparklines) — same rule as ShotHeatmap: distinct player_ids ascending,
  // first = --player1-color. Union of ids from tips and report-stats so a
  // player who only shows up in one of the two still gets a stable color.
  const statsPlayerIds = stats
    ? [
        ...Object.keys(stats.shot_counts || {}),
        ...Object.values(stats.rallies || {}).flatMap((r) => Object.keys(r.kitchen_arrival_s || {})),
      ]
    : []
  const colorMap = buildPlayerColorMap([
    ...tips.map((t) => t.player_id),
    ...statsPlayerIds.map(Number),
  ])

  return (
    <div className="rally-analysis-panel">
      <div className="row spread">
        <h3 style={{ margin: 0 }}>Rally Analysis</h3>
        <CoachingSetupSelector video={video} onUpdateVideo={onUpdateVideo} />
      </div>
      {video?.coaching_focus === 'positioning' && !video?.match_type && (
        <div className="muted" style={{ margin: '4px 0 8px' }}>
          Tip: set Singles or Doubles above — positioning feedback uses it to enable the
          recovery (singles) and partner-spacing (doubles) checks.
        </div>
      )}

      <CoachingSelections videoId={videoId} fps={video?.fps || 30} onSeek={onSeek} frame={frame} />

      <div className="panel">
        <div className="row spread">
          <h2>Coaching</h2>
          <div className="row">
            <button onClick={onProcessSingleRally} disabled={markingSingleRally}
                    title="For a pre-trimmed clip that is exactly one rally: skip rally detection and process the whole clip as a single rally.">
              {markingSingleRally ? 'Submitting…' : 'Whole clip = one rally'}
            </button>
            <button onClick={onGenerateReport} disabled={generating}>
              {generating ? 'Generating…' : report ? 'Regenerate report' : 'Generate report'}
            </button>
          </div>
        </div>

        {!report ? (
          <div className="clip-list-empty">No coaching report yet.</div>
        ) : (
          <>
            {report.stale_reason && (
              <div className="stale-report-banner">
                ⚠️ This report was generated before a data-accuracy fix and may contain
                unreliable feedback — regenerate for accurate results.
              </div>
            )}
            {(report.focus || 'form') !== (video?.coaching_focus || 'form') && (
              <div className="stale-report-banner">
                ⚠️ This report gives {report.focus || 'form'} feedback, but this clip is now
                set to {video?.coaching_focus || 'form'} — regenerate to match.
              </div>
            )}
            {players.length > 1 && (
              <div className="tabs" style={{ margin: '4px 0 8px' }}>
                <button className={selectedPlayer === ALL_PLAYERS ? 'active' : ''}
                        onClick={() => setSelectedPlayer(ALL_PLAYERS)}>
                  All players
                </button>
                {players.map((p) => {
                  const color = playerColor(colorMap, labelToPlayerId(p))
                  return (
                    <button key={p} className={selectedPlayer === p ? 'active' : ''}
                            style={selectedPlayer === p ? { borderColor: color, color } : undefined}
                            onClick={() => setSelectedPlayer(p)}>
                      {p}
                    </button>
                  )
                })}
              </div>
            )}
            {visibleGeneral.length > 0 ? (
              <div className="theme-cards">
                {visibleGeneral.map((g) => (
                  <div key={g.player_label} className="theme-card">
                    <span className="theme-card-player"
                          style={{ color: playerColor(colorMap, g.player_id) }}>
                      {g.player_label}
                    </span>
                    <p className="theme-card-text">👍 <strong>What's working:</strong> {g.praise}</p>
                    <p className="theme-card-text">🎯 <strong>Focus on:</strong> {g.improve}</p>
                  </div>
                ))}
              </div>
            ) : visibleThemes.length > 0 && (
              <div className="theme-cards">
                {visibleThemes.map((t) => (
                  <div key={t.player_label} className="theme-card">
                    <span className="theme-card-player"
                          style={{ color: playerColor(colorMap, t.player_id) }}>
                      {t.player_label} — session focus
                    </span>
                    <p className="theme-card-text">{t.theme}</p>
                  </div>
                ))}
              </div>
            )}
            <CoachingVideo videoId={videoId} tips={tips} selectedPlayer={selectedPlayer}
                           selectedTip={selectedTip} onSelectTip={setSelectedTip} colorMap={colorMap}
                           tipFeedback={tipFeedback} onRateTip={onRateTip} />
          </>
        )}
      </div>

      <div className="panel">
        <h2>Report Stats</h2>
        {stats === null && !statsError ? (
          <div className="clip-list-empty">Loading stats…</div>
        ) : statsError ? (
          <div className="error">Failed to load report stats.</div>
        ) : (
          <ReportStats stats={stats} colorMap={colorMap} />
        )}
      </div>

      <div className="panel">
        <ShotHeatmap videoId={videoId} />
      </div>
    </div>
  )
}

// Self-contained video + pin scrubber, independent of the shared Workspace
// player (separate <video> element, own play/seek state) so the coaching
// view can be scrubbed/pinned without disturbing Setup/Pass tab playback.
function CoachingVideo({ videoId, tips, selectedPlayer, selectedTip, onSelectTip, colorMap,
                          tipFeedback, onRateTip }) {
  const videoRef = useRef(null)
  const [duration, setDuration] = useState(0)

  const visibleTips = selectedPlayer === ALL_PLAYERS
    ? tips
    : tips.filter((t) => t.player_label === selectedPlayer)

  const seekToTip = (t) => {
    const v = videoRef.current
    if (v) v.currentTime = t.timestamp_s
    onSelectTip(t)
  }

  const onScrubClick = (e) => {
    const v = videoRef.current
    if (!v || !duration) return
    const rect = e.currentTarget.getBoundingClientRect()
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    v.currentTime = frac * duration
  }

  const blocks = visibleTips.map((t) => ({
    id: t.contact_id,
    start: Math.max(0, t.timestamp_s - PRE_WINDOW_S),
    end: Math.min(duration || t.timestamp_s + POST_WINDOW_S, t.timestamp_s + POST_WINDOW_S),
  }))
  const { laneById, laneCount } = assignLanes(blocks)
  const trackHeight = LANE_ROW_GAP_PX + laneCount * (LANE_HEIGHT_PX + LANE_ROW_GAP_PX)

  return (
    <div className="rally-video">
      <video ref={videoRef} src={api.streamUrl(videoId)} controls preload="metadata"
             onLoadedMetadata={(e) => setDuration(e.target.duration || 0)} />
      <div className="pin-track" style={{ height: `${trackHeight}px` }} onClick={onScrubClick}>
        {visibleTips.map((t) => {
          const start = Math.max(0, t.timestamp_s - PRE_WINDOW_S)
          const end = Math.min(duration || t.timestamp_s + POST_WINDOW_S, t.timestamp_s + POST_WINDOW_S)
          const { lane, overflow } = laneById.get(t.contact_id) || { lane: 0, overflow: false }
          return (
            <button key={t.contact_id} type="button"
                    className={`pin tip-${t.tone}${selectedTip?.contact_id === t.contact_id ? ' pin-active' : ''}`}
                    style={{
                      left: `${duration ? (start / duration) * 100 : 0}%`,
                      width: `${duration ? ((end - start) / duration) * 100 : 0}%`,
                      top: `${LANE_ROW_GAP_PX + lane * (LANE_HEIGHT_PX + LANE_ROW_GAP_PX)}px`,
                      height: `${LANE_HEIGHT_PX}px`,
                      background: playerColor(colorMap, t.player_id),
                      // Overflow (beyond MAX_LANES) blocks draw more transparent to
                      // signal "several more here" -- unless this one is the active
                      // selection, which must stay fully opaque (inline style would
                      // otherwise win over the .pin-active CSS rule).
                      opacity: overflow && selectedTip?.contact_id !== t.contact_id ? 0.4 : undefined,
                    }}
                    title={`${t.player_label} · ${formatCategory(t.category)} · ${formatTime(t.timestamp_s)}`}
                    onClick={(e) => { e.stopPropagation(); seekToTip(t) }} />
          )
        })}
      </div>

      {selectedTip ? (
        <div className={`tip-panel tip-${selectedTip.tone}`}>
          <div className="row spread">
            <span className="tip-card-badge">
              {selectedTip.tone === 'praise' ? '👍' : '🎯'} {formatCategory(selectedTip.category)}
            </span>
            <span className="tip-panel-player" style={{ color: playerColor(colorMap, selectedTip.player_id) }}>
              {selectedTip.player_label}
            </span>
          </div>
          <p className="tip-card-text">{selectedTip.tip}</p>
          <div className="tip-feedback">
            <span className="tip-feedback-label">Was this tip helpful?</span>
            <button
              className={`tip-feedback-btn${tipFeedback?.[selectedTip.contact_id] === 1 ? ' active' : ''}`}
              title="Helpful"
              onClick={() => onRateTip?.(selectedTip, 1)}>
              👍
            </button>
            <button
              className={`tip-feedback-btn${tipFeedback?.[selectedTip.contact_id] === -1 ? ' active' : ''}`}
              title="Not helpful"
              onClick={() => onRateTip?.(selectedTip, -1)}>
              👎
            </button>
          </div>
        </div>
      ) : (
        <div className="clip-list-empty">
          {visibleTips.length ? "Click a pin to see that shot's feedback." : 'No tips for this player.'}
        </div>
      )}
    </div>
  )
}

const ZONES = ['kitchen', 'midcourt', 'backcourt']
const SIDES = ['left', 'right']

// Shot counts by side/zone, kitchen arrival per rally, and a forward-momentum
// sparkline per rally — from GET .../report-stats. Nulls (missing data) are
// always shown as "—", never coerced to 0.
function ReportStats({ stats, colorMap }) {
  const shotCounts = stats.shot_counts || {}
  const playerIds = Object.keys(shotCounts).map(Number).sort((a, b) => a - b)
  const rallyIds = Object.keys(stats.rallies || {})

  if (playerIds.length === 0 && rallyIds.length === 0) {
    return <div className="clip-list-empty">No rally stats yet.</div>
  }

  // Compact per-player kitchen-arrival summary (avg across rallies), shown by
  // default; the full per-rally breakdown (IDs, frames, sparklines) lives
  // behind a single collapsed disclosure below.
  const kitchenArrivals = {}
  rallyIds.forEach((rid) => {
    const rally = stats.rallies[rid]
    Object.entries(rally.kitchen_arrival_s || {}).forEach(([pid, arrival]) => {
      if (arrival == null) return
      ;(kitchenArrivals[pid] || (kitchenArrivals[pid] = [])).push(arrival)
    })
  })
  const summaryPlayerIds = Object.keys(kitchenArrivals).map(Number).sort((a, b) => a - b)

  return (
    <div>
      {playerIds.length > 0 && (
        <table className="rally-stats-table">
          <thead>
            <tr>
              <th>Player</th>
              {SIDES.flatMap((side) => ZONES.map((zone) => <th key={`${side}-${zone}`}>{side}/{zone}</th>))}
            </tr>
          </thead>
          <tbody>
            {playerIds.map((pid) => (
              <tr key={pid}>
                <td style={{ color: playerColor(colorMap, pid), fontWeight: 600 }}>Player {pid}</td>
                {SIDES.flatMap((side) => ZONES.map((zone) => {
                  const c = shotCounts[pid]?.[side]?.[zone]
                  const total = c ? c.serve + c.non_serve : 0
                  return (
                    <td key={`${side}-${zone}`}>
                      {c ? `${total}${c.serve ? ` (${c.serve}s)` : ''}` : <span className="rally-stats-empty">–</span>}
                    </td>
                  )
                }))}
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {summaryPlayerIds.length > 0 && (
        <div className="rally-stats-summary">
          {summaryPlayerIds.map((pid) => {
            const arrivals = kitchenArrivals[pid]
            const avg = arrivals.reduce((a, b) => a + b, 0) / arrivals.length
            return (
              <div key={pid} className="rally-stats-summary-row">
                <span style={{ color: playerColor(colorMap, pid), fontWeight: 600 }}>Player {pid}</span>
                <span className="muted">
                  avg kitchen arrival {avg.toFixed(1)}s over {arrivals.length} rally{arrivals.length === 1 ? '' : 's'}
                </span>
              </div>
            )
          })}
        </div>
      )}

      {rallyIds.length > 0 && (
        <details className="rally-stats-raw">
          <summary>Raw rally data</summary>
          {rallyIds.map((rid) => {
            const rally = stats.rallies[rid]
            const rallyPlayerIds = Object.keys(rally.kitchen_arrival_s || {}).map(Number).sort((a, b) => a - b)
            return (
              <div key={rid} className="rally-stats-block">
                <div className="rally-stats-block-title">Rally {rid} (frames {rally.start_frame}–{rally.end_frame})</div>
                <div className="row" style={{ gap: 24, alignItems: 'flex-start' }}>
                  {rallyPlayerIds.map((pid) => {
                    const arrival = rally.kitchen_arrival_s?.[pid]
                    const series = rally.forward_momentum?.[pid] || []
                    return (
                      <div key={pid}>
                        <div style={{ color: playerColor(colorMap, pid), fontWeight: 600, fontSize: 12 }}>
                          Player {pid}
                        </div>
                        <div className="muted">
                          Kitchen arrival: {arrival != null ? `${arrival.toFixed(1)}s` : <span className="rally-stats-empty">–</span>}
                        </div>
                        <Sparkline series={series} color={playerColor(colorMap, pid)} />
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </details>
      )}
    </div>
  )
}

// Minimal inline SVG line of court_y (net-relative depth) over frames — no
// chart library, just a normalized polyline. Frames with no position sample
// are skipped entirely (never 0-filled).
function Sparkline({ series, color }) {
  const w = 120, h = 30
  const values = series.map((s) => s.court_y).filter((v) => v != null)
  if (values.length < 2) return <div className="muted" style={{ fontSize: 12 }}>No position data.</div>
  const min = Math.min(...values)
  const range = Math.max(...values) - min || 1
  const points = values
    .map((v, i) => `${(i / (values.length - 1)) * w},${h - ((v - min) / range) * h}`)
    .join(' ')
  return (
    <svg width={w} height={h} className="sparkline">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  )
}

function formatCategory(category) {
  return (category || '').replace(/_/g, ' ')
}

function formatTime(s) {
  if (s == null) return ''
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}
