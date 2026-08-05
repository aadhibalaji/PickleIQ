import React, { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'

// Pass 1 candidate review + manual range drawing, feeding the coaching
// pipeline below: accept/skip each Pass 1 rally (or draw a range Pass 1
// missed), then process the selection into overlays coaching-report/
// report-stats/trajectories read from. Lives at the top of
// RallyAnalysisPanel since it's the step that makes those useful.
export default function CoachingSelections({ videoId, fps = 30, onSeek }) {
  const [candidates, setCandidates] = useState([])
  const [choice, setChoice] = useState({}) // rally_label_id -> 'include' | 'skip'
  const [drawn, setDrawn] = useState([]) // [{start_frame, end_frame}]
  const [newRange, setNewRange] = useState({ start_frame: '', end_frame: '' })
  const [selections, setSelections] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const loadCandidates = useCallback(() => {
    api.getPass1Candidates(videoId).then((r) => {
      setCandidates(r.candidates)
      setChoice((c) => {
        const next = { ...c }
        for (const cand of r.candidates) if (!(cand.id in next)) next[cand.id] = 'skip'
        return next
      })
    }).catch((e) => setError(e.message))
  }, [videoId])

  const loadSelections = useCallback(() => {
    api.listSelections(videoId).then((r) => setSelections(r.selections)).catch(() => {})
  }, [videoId])

  useEffect(() => { loadCandidates(); loadSelections() }, [loadCandidates, loadSelections])

  // Same polling pattern as Workspace.jsx's job-status poll — just against
  // the selections endpoint instead of /videos/{id}.
  useEffect(() => {
    const active = selections.some((s) => s.status === 'pending' || s.status === 'processing')
    if (!active) return
    const t = setInterval(loadSelections, 3000)
    return () => clearInterval(t)
  }, [selections, loadSelections])

  const selectAll = () => {
    setChoice((c) => {
      const next = { ...c }
      for (const cand of candidates) next[cand.id] = 'include'
      return next
    })
  }

  const addDrawnRange = () => {
    const start = parseInt(newRange.start_frame, 10)
    const end = parseInt(newRange.end_frame, 10)
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      setError('enter a valid start/end frame (end must be after start)')
      return
    }
    setError('')
    setDrawn((d) => [...d, { start_frame: start, end_frame: end }])
    setNewRange({ start_frame: '', end_frame: '' })
  }

  const removeDrawnRange = (idx) => setDrawn((d) => d.filter((_, i) => i !== idx))

  const includedCandidates = candidates.filter((c) => choice[c.id] === 'include')
  const hasSelection = includedCandidates.length > 0 || drawn.length > 0

  const processSelected = async () => {
    if (!hasSelection || submitting) return
    setSubmitting(true)
    setError('')
    try {
      const items = [
        ...includedCandidates.map((c) => ({ source: 'pass1_accepted', rally_label_id: c.id })),
        ...drawn.map((r) => ({ source: 'user_drawn', start_frame: r.start_frame, end_frame: r.end_frame })),
      ]
      const { created } = await api.createSelections(videoId, items)
      for (const selectionId of created) {
        await api.processSelection(videoId, selectionId)
      }
      setDrawn([])
      loadSelections()
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="panel">
      <div className="row spread">
        <h2 style={{ margin: 0 }}>Coaching Selections</h2>
        <div className="row">
          <button onClick={selectAll} disabled={candidates.length === 0}>Select all Pass 1</button>
          <button className="blue" onClick={processSelected} disabled={!hasSelection || submitting}
                  title={hasSelection ? undefined : 'Include at least one rally or add a range first'}>
            {submitting ? 'Processing…' : 'Process selected'}
          </button>
        </div>
      </div>
      <div className="muted" style={{ marginBottom: 8 }}>
        Accept the Pass 1 rallies to use for coaching analysis, or draw a range Pass 1
        missed. Nothing runs until you click "Process selected".
      </div>

      <table>
        <thead><tr><th>#</th><th>Start</th><th>End</th><th>Length</th><th></th><th></th></tr></thead>
        <tbody>
          {candidates.map((c, i) => (
            <tr key={c.id}>
              <td>{i + 1}</td>
              <td>{fmt(c.start_frame, fps)}</td>
              <td>{fmt(c.end_frame, fps)}</td>
              <td>{((c.end_frame - c.start_frame) / fps).toFixed(1)}s</td>
              <td>
                <button className={choice[c.id] === 'include' ? 'active' : ''}
                        style={choice[c.id] === 'include' ? { borderColor: '#4cc38a', color: '#4cc38a' } : undefined}
                        onClick={() => setChoice((ch) => ({ ...ch, [c.id]: 'include' }))}>
                  Include
                </button>
                {' '}
                <button className={choice[c.id] === 'skip' ? 'active' : ''}
                        onClick={() => setChoice((ch) => ({ ...ch, [c.id]: 'skip' }))}>
                  Skip
                </button>
              </td>
              <td>{onSeek && <button onClick={() => onSeek(c.start_frame)}>▶</button>}</td>
            </tr>
          ))}
          {candidates.length === 0 && (
            <tr><td colSpan="6" className="muted">No Pass 1 candidates found yet.</td></tr>
          )}
        </tbody>
      </table>

      <div className="row spread" style={{ marginTop: 10, alignItems: 'flex-end' }}>
        <div className="row">
          <span className="muted">Add a missed range:</span>
          <input type="number" placeholder="start frame" style={{ width: 110 }} value={newRange.start_frame}
                 onChange={(e) => setNewRange((r) => ({ ...r, start_frame: e.target.value }))} />
          <input type="number" placeholder="end frame" style={{ width: 110 }} value={newRange.end_frame}
                 onChange={(e) => setNewRange((r) => ({ ...r, end_frame: e.target.value }))} />
          <button onClick={addDrawnRange}>Add range</button>
        </div>
      </div>
      {drawn.map((r, i) => (
        <div key={i} className="row spread" style={{ padding: '4px 0' }}>
          <span className="muted">Frames {fmt(r.start_frame, fps)} – {fmt(r.end_frame, fps)} (drawn)</span>
          <button className="danger" onClick={() => removeDrawnRange(i)}>Remove</button>
        </div>
      ))}

      {error && <div className="error">{error}</div>}

      {selections.length > 0 && (
        <>
          <div className="muted" style={{ marginTop: 10 }}>Selections</div>
          {selections.map((s) => (
            <div key={s.id} className="row spread" style={{ padding: '4px 0' }}>
              <span>{fmt(s.start_frame, fps)} – {fmt(s.end_frame, fps)} ({s.source})</span>
              <span className={s.status === 'failed' ? 'error' : 'muted'}>
                {s.status}{s.status === 'failed' && s.error_message ? `: ${s.error_message}` : ''}
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  )
}

function fmt(f, fps) {
  const s = f / fps
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')} (${f})`
}
