import React, { useMemo, useState } from 'react'
import { api } from './api.js'

/*
Manual rally labeling with a strict alternating state machine:
after Start only End is enabled; after End only Start. Malformed
label pairs are impossible by construction.

Labeling definition (matches the system's definition exactly):
  START = the serve (players set, first ball motion)
  END   = the frame the ball becomes UNPLAYABLE (bounce out, second
          bounce, net fail) — NOT when the ball stops rolling and NOT
          when players reset.
*/
export default function LabelingPanel({ videoId, fps, frame, labels, setLabels, onSeek, onSaved }) {
  const [pendingStart, setPendingStart] = useState(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const sorted = useMemo(
    () => labels.map((l, i) => ({ ...l, i })).sort((a, b) => a.start_frame - b.start_frame),
    [labels],
  )

  const clickStart = () => setPendingStart(frame)
  const clickEnd = () => {
    if (pendingStart === null || frame <= pendingStart) {
      setError('End must be after start')
      return
    }
    setError('')
    setLabels([...labels, { start_frame: pendingStart, end_frame: frame }])
    setPendingStart(null)
  }

  const adjust = (i, field) => {
    const next = labels.map((l, j) => (j === i ? { ...l, [field]: frame } : l))
    const l = next[i]
    if (l.end_frame <= l.start_frame) { setError('End must stay after start'); return }
    setError('')
    setLabels(next)
  }
  const remove = (i) => setLabels(labels.filter((_, j) => j !== i))

  const save = async () => {
    setSaving(true)
    setError('')
    try {
      await api.putHumanLabels(videoId, labels)
      onSaved?.()
    } catch (e) {
      setError(e.message)
    }
    setSaving(false)
  }

  return (
    <div className="panel">
      <h2>Rally labels (ground truth)</h2>
      <div className="muted" style={{ marginBottom: 8 }}>
        Mark <b>Start</b> at the serve. Mark <b>End</b> at the exact frame the ball becomes{' '}
        <b>unplayable</b> — bounces out, bounces twice, or dies at the net. Not when the ball
        stops rolling, not when players reset. Consistent ends are what make the comparison meaningful.
      </div>
      <div className="row">
        <button className="primary" onClick={clickStart} disabled={pendingStart !== null}>
          ▶ Start rally @ f{frame}
        </button>
        <button className="danger" onClick={clickEnd} disabled={pendingStart === null}>
          ■ End rally @ f{frame}
        </button>
        {pendingStart !== null && (
          <span className="muted">
            rally open since f{pendingStart} ({(pendingStart / fps).toFixed(1)}s)
            {' '}<button onClick={() => setPendingStart(null)}>cancel</button>
          </span>
        )}
        <span style={{ flex: 1 }} />
        <button className="blue" onClick={save} disabled={saving}>
          {saving ? 'Saving…' : `Save ${labels.length} labels`}
        </button>
      </div>
      {error && <div className="error">{error}</div>}

      {sorted.length > 0 && (
        <table style={{ marginTop: 10 }}>
          <thead>
            <tr><th>#</th><th>Start</th><th>End</th><th>Len</th><th>Review / adjust</th></tr>
          </thead>
          <tbody>
            {sorted.map((l, k) => (
              <tr key={l.i}>
                <td>{k + 1}</td>
                <td>{(l.start_frame / fps).toFixed(1)}s ({l.start_frame})</td>
                <td>{(l.end_frame / fps).toFixed(1)}s ({l.end_frame})</td>
                <td>{((l.end_frame - l.start_frame) / fps).toFixed(1)}s</td>
                <td className="row">
                  <button onClick={() => onSeek(l.start_frame)}>▶ start</button>
                  <button onClick={() => onSeek(l.end_frame)}>▶ end</button>
                  <button onClick={() => adjust(l.i, 'start_frame')}>start = f{frame}</button>
                  <button onClick={() => adjust(l.i, 'end_frame')}>end = f{frame}</button>
                  <button className="danger" onClick={() => remove(l.i)}>✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
