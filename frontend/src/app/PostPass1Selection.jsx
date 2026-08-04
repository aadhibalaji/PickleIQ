import React, { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'

// First real screen of the new user-facing flow (app/): after Pass 1 finds
// candidate rallies for a video, the user accepts/skips each one and can
// draw extra ranges Pass 1 missed, then kicks off Pass 2 for whatever was
// selected. Deliberately does NOT touch Workspace.jsx/its children — this is
// a separate, simpler surface over the same coaching_selections endpoints
// (see api/rally_analysis.py).
export default function PostPass1Selection({ videoId }) {
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

  // Poll while any selection is still pending/processing — same idea as
  // Workspace.jsx's job-status poll (see its "Poll jobs + streamed results"
  // effect), just against the selections endpoint instead of /videos/{id}.
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
    <div>
      <div className="row spread">
        <h2 style={{ margin: 0 }}>Review Pass 1 rallies</h2>
        <button onClick={selectAll} disabled={candidates.length === 0}>Select all</button>
      </div>

      {candidates.length === 0 && <div className="muted">No Pass 1 candidates found yet.</div>}
      {candidates.map((c) => (
        <div key={c.id} className="panel row spread" style={{ padding: 10 }}>
          <span>Frames {c.start_frame}–{c.end_frame}</span>
          <div className="row">
            <button
              className={choice[c.id] === 'include' ? 'blue' : ''}
              onClick={() => setChoice((ch) => ({ ...ch, [c.id]: 'include' }))}
            >
              Include
            </button>
            <button
              className={choice[c.id] === 'skip' ? 'active' : ''}
              onClick={() => setChoice((ch) => ({ ...ch, [c.id]: 'skip' }))}
            >
              Skip
            </button>
          </div>
        </div>
      ))}

      <h3>Add a missed range</h3>
      <div className="panel row" style={{ padding: 10 }}>
        <input type="number" placeholder="start frame" value={newRange.start_frame}
               onChange={(e) => setNewRange((r) => ({ ...r, start_frame: e.target.value }))} />
        <input type="number" placeholder="end frame" value={newRange.end_frame}
               onChange={(e) => setNewRange((r) => ({ ...r, end_frame: e.target.value }))} />
        <button onClick={addDrawnRange}>Add range</button>
      </div>
      {drawn.map((r, i) => (
        <div key={i} className="row spread" style={{ padding: '4px 10px' }}>
          <span className="muted">Frames {r.start_frame}–{r.end_frame} (drawn)</span>
          <button className="danger" onClick={() => removeDrawnRange(i)}>Remove</button>
        </div>
      ))}

      {error && <div className="error">{error}</div>}

      <div className="row" style={{ marginTop: 12 }}>
        <button className="blue" onClick={processSelected} disabled={!hasSelection || submitting}>
          {submitting ? 'Processing…' : 'Process selected'}
        </button>
        {!hasSelection && (
          <span className="muted">Include at least one rally or add a range before processing.</span>
        )}
      </div>

      {selections.length > 0 && (
        <>
          <h3>Selections</h3>
          {selections.map((s) => (
            <div key={s.id} className="row spread" style={{ padding: '4px 10px' }}>
              <span>Frames {s.start_frame}–{s.end_frame} ({s.source})</span>
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
