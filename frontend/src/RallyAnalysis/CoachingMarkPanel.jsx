import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'

/**
 * Expert labeling of coaching moments — the ground truth every threshold in
 * frame_metrics is calibrated against (backend/scripts/eval_coaching.py,
 * calibrate_thresholds.py's supervised sweep).
 *
 * TWO MODES, and both are required for the numbers to mean anything:
 *
 *   Review    steps through the moments the detector ALREADY found and asks
 *             "was this actually bad?" -> PRECISION.
 *   Free      lets you mark any span yourself, whether or not anything was
 *             flagged there -> RECALL. This is the only mode that can reveal
 *             what the detector is blind to, so a session that only reviews
 *             spikes can never show a miss.
 *
 * Reviewed spikes are keyed by their contact_id, so re-marking one replaces
 * the earlier verdict rather than double-counting it.
 */

const VERDICTS = [
  ['bad', '1', 'Bad — a real problem worth coaching'],
  ['fine', '2', 'Fine — nothing wrong here'],
  ['good', '3', 'Good — actively well done'],
]

/**
 * Which metric drove this moment — the one with the largest |deviation| at the
 * peak frame. Recorded on the mark so eval_coaching can attribute false
 * positives to a specific metric instead of blaming the focus as a whole.
 * Derived here rather than read off the payload because debug_timeline reports
 * every metric's deviation, not a single dominant one.
 */
function dominantMetric(shot) {
  const entries = Object.entries(shot?.metrics || {})
    .filter(([, m]) => m && m.deviation != null)
  if (!entries.length) return undefined
  return entries.reduce((best, e) =>
    Math.abs(e[1].deviation) > Math.abs(best[1].deviation) ? e : best)[0]
}

export default function CoachingMarkPanel({ videoId, video, onSeek, frame }) {
  const focus = video?.coaching_focus || 'form'
  const fps = video?.fps || 30

  const [mode, setMode] = useState('review')
  const [spikes, setSpikes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [marks, setMarks] = useState([])
  const [idx, setIdx] = useState(0)
  // Freeform span under construction: [ sets the start, ] the end. Null until
  // the first bound is set, so an accidental verdict keypress can't save a
  // zero-length span at whatever frame happened to be showing.
  const [span, setSpan] = useState(null)
  const [playerId, setPlayerId] = useState(null)
  const [note, setNote] = useState('')

  const loadMarks = useCallback(() => {
    api.listCoachingMarks(videoId, focus)
      .then((r) => setMarks(r.marks || []))
      .catch(() => setMarks([]))
  }, [videoId, focus])

  useEffect(() => { loadMarks() }, [loadMarks])
  useEffect(() => { setSpikes(null); setIdx(0); setSpan(null) }, [videoId, focus])

  const loadSpikes = () => {
    setLoading(true)
    setError(null)
    api.getCoachingDebug(videoId, focus)
      .then((d) => {
        // Only span moments are labelable: they carry the frame bounds the
        // marks table stores. Sorted by time so review order matches playback.
        const found = (d.shots || [])
          .filter((s) => s.start_frame != null && s.end_frame != null)
          .sort((a, b) => (a.timestamp_s || 0) - (b.timestamp_s || 0))
        setSpikes(found)
        setIdx(0)
        if (found.length) onSeek?.(found[0].start_frame)
      })
      .catch(() => setError('Could not load detected moments — has Pass 2 finished for this clip?'))
      .finally(() => setLoading(false))
  }

  const current = spikes && spikes.length ? spikes[Math.min(idx, spikes.length - 1)] : null
  const marksBySpike = useMemo(() => {
    const m = {}
    marks.forEach((mk) => { if (mk.spike_id) m[mk.spike_id] = mk })
    return m
  }, [marks])
  const reviewedCount = spikes ? spikes.filter((s) => marksBySpike[s.contact_id]).length : 0
  const freeformMarks = marks.filter((m) => m.source === 'freeform')
  const badFreeform = freeformMarks.filter((m) => m.verdict === 'bad').length

  const goTo = useCallback((next) => {
    if (!spikes || !spikes.length) return
    const clamped = Math.max(0, Math.min(spikes.length - 1, next))
    setIdx(clamped)
    onSeek?.(spikes[clamped].start_frame)
  }, [spikes, onSeek])

  const saveSpikeVerdict = useCallback((verdict) => {
    if (!current) return
    api.addCoachingMark(videoId, {
      player_id: current.player_id, focus,
      start_frame: current.start_frame, end_frame: current.end_frame,
      verdict, source: 'spike', spike_id: current.contact_id,
      rally_id: current.rally_id, metric: dominantMetric(current),
      note: note || undefined,
    }).then(() => {
      setNote('')
      loadMarks()
      goTo(idx + 1)   // advance automatically: reviewing is a rhythm, not a form
    }).catch(() => setError('Save failed'))
  }, [current, videoId, focus, note, loadMarks, goTo, idx])

  const saveFreeform = useCallback((verdict) => {
    if (!span || span.start == null || span.end == null) return
    if (playerId == null) { setError('Pick which player this is about first'); return }
    api.addCoachingMark(videoId, {
      player_id: playerId, focus,
      start_frame: Math.min(span.start, span.end),
      end_frame: Math.max(span.start, span.end),
      verdict, source: 'freeform', note: note || undefined,
    }).then(() => {
      setSpan(null)
      setNote('')
      loadMarks()
    }).catch(() => setError('Save failed'))
  }, [span, playerId, videoId, focus, note, loadMarks])

  // Keyboard, matching the ball-marking tool's conventions (Workspace.jsx):
  // never fires while typing in a field, and arrows step frames.
  useEffect(() => {
    const onKey = (e) => {
      const t = e.target
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      const k = e.key.toLowerCase()
      const verdict = VERDICTS.find(([, key]) => key === e.key)
      if (verdict) {
        e.preventDefault()
        return mode === 'review' ? saveSpikeVerdict(verdict[0]) : saveFreeform(verdict[0])
      }
      if (mode === 'review') {
        if (k === 'j' || e.key === 'ArrowDown') { e.preventDefault(); goTo(idx + 1) }
        else if (k === 'k' || e.key === 'ArrowUp') { e.preventDefault(); goTo(idx - 1) }
      } else {
        if (e.key === '[') { e.preventDefault(); setSpan((s) => ({ ...(s || {}), start: frame })) }
        else if (e.key === ']') { e.preventDefault(); setSpan((s) => ({ ...(s || {}), end: frame })) }
        else if (e.key === 'Escape') { e.preventDefault(); setSpan(null) }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mode, idx, frame, goTo, saveSpikeVerdict, saveFreeform])

  const playerOptions = useMemo(() => {
    const ids = new Set(marks.map((m) => m.player_id))
    ;(spikes || []).forEach((s) => ids.add(s.player_id))
    return [...ids].filter((v) => v != null).sort((a, b) => a - b)
  }, [marks, spikes])

  return (
    <div className="coaching-mark-panel">
      <div className="row spread">
        <strong>Label moments ({focus})</strong>
        <span className="muted">
          {marks.length} marks · {badFreeform} bad spans found by you
        </span>
      </div>
      <p className="muted" style={{ margin: '4px 0 8px' }}>
        These labels are what thresholds get calibrated against. Reviewing flagged
        moments measures how often the detector is right; marking your own spans
        measures what it misses — both are needed. Aim for 100+ marks.
      </p>

      <div className="row" style={{ gap: 8 }}>
        <button className={mode === 'review' ? 'active' : ''} onClick={() => setMode('review')}>
          Review flagged
        </button>
        <button className={mode === 'free' ? 'active' : ''} onClick={() => setMode('free')}>
          Mark my own
        </button>
      </div>

      {error && <div className="muted" style={{ color: 'var(--danger, #c00)' }}>{error}</div>}

      {mode === 'review' && (
        <div style={{ marginTop: 8 }}>
          {!spikes && (
            <button onClick={loadSpikes} disabled={loading}>
              {loading ? 'Loading…' : 'Load detected moments'}
            </button>
          )}
          {spikes && !spikes.length && (
            <div className="muted">
              Nothing was flagged for this clip — switch to “Mark my own” and record the
              moments that should have been caught.
            </div>
          )}
          {current && (
            <>
              <div className="row spread" style={{ marginTop: 4 }}>
                <span>Moment {idx + 1} / {spikes.length} · reviewed {reviewedCount}</span>
                <span>
                  <button onClick={() => goTo(idx - 1)} disabled={idx === 0}>← prev (k)</button>
                  {' '}
                  <button onClick={() => goTo(idx + 1)} disabled={idx >= spikes.length - 1}>
                    next (j) →
                  </button>
                </span>
              </div>
              <div className="muted" style={{ margin: '4px 0' }}>
                Player {current.player_id} · {current.category} ·{' '}
                {current.metrics ? Object.keys(current.metrics).length : 0} metrics ·{' '}
                frames {current.start_frame}–{current.end_frame}
                {' '}<button onClick={() => onSeek?.(current.start_frame)}>▶ replay</button>
              </div>
              {marksBySpike[current.contact_id] && (
                <div className="muted">
                  Already marked: <strong>{marksBySpike[current.contact_id].verdict}</strong>
                  {' '}(marking again replaces it)
                </div>
              )}
            </>
          )}
        </div>
      )}

      {mode === 'free' && (
        <div style={{ marginTop: 8 }}>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <label>
              Player{' '}
              <select value={playerId ?? ''} onChange={(e) => setPlayerId(Number(e.target.value))}>
                <option value="" disabled>pick…</option>
                {playerOptions.map((p) => <option key={p} value={p}>Player {p}</option>)}
              </select>
            </label>
            <button onClick={() => setSpan((s) => ({ ...(s || {}), start: frame }))}>
              [ set start
            </button>
            <button onClick={() => setSpan((s) => ({ ...(s || {}), end: frame }))}>
              ] set end
            </button>
            <span className="muted">
              {span?.start != null ? `start ${span.start}` : 'start —'} ·{' '}
              {span?.end != null ? `end ${span.end}` : 'end —'}
            </span>
          </div>
          {playerOptions.length === 0 && (
            <div className="muted">
              No players known yet — load the detected moments once in “Review flagged”
              to populate this list.
            </div>
          )}
        </div>
      )}

      <div className="row" style={{ gap: 8, marginTop: 8, alignItems: 'center' }}>
        {VERDICTS.map(([v, key, title]) => (
          <button key={v} title={title}
                  disabled={mode === 'review' ? !current : !(span?.start != null && span?.end != null)}
                  onClick={() => (mode === 'review' ? saveSpikeVerdict(v) : saveFreeform(v))}>
            {v} ({key})
          </button>
        ))}
        <input placeholder="optional note" value={note} style={{ flex: 1 }}
               onChange={(e) => setNote(e.target.value)} />
      </div>

      {marks.length > 0 && (
        <table className="compact" style={{ marginTop: 10, width: '100%' }}>
          <thead>
            <tr><th>player</th><th>frames</th><th>verdict</th><th>source</th><th>note</th><th /></tr>
          </thead>
          <tbody>
            {marks.map((m) => (
              <tr key={m.id}>
                <td>{m.player_id}</td>
                <td>
                  <button onClick={() => onSeek?.(m.start_frame)}>▶</button>{' '}
                  {m.start_frame}–{m.end_frame}
                </td>
                <td>{m.verdict}</td>
                <td className="muted">{m.source}</td>
                <td className="muted">{m.note}</td>
                <td>
                  <button onClick={() => api.deleteCoachingMark(m.id).then(loadMarks)}>✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
