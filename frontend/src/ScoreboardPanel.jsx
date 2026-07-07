import React, { useEffect, useState } from 'react'
import { api } from './api.js'

// Tiny inline sparkline (no chart lib): plot one series over version index.
function Spark({ vals, color, w, h, min, max }) {
  const pts = vals.filter((v) => v != null)
  if (pts.length < 2) return null
  const span = max - min || 1
  const dx = w / (vals.length - 1)
  const d = vals
    .map((v, i) => (v == null ? null : `${i === 0 ? 'M' : 'L'}${(i * dx).toFixed(1)},${(h - ((v - min) / span) * h).toFixed(1)}`))
    .filter(Boolean)
    .join(' ')
  return <path d={d} fill="none" stroke={color} strokeWidth="2" />
}

export default function ScoreboardPanel({ refreshKey }) {
  const [sb, setSb] = useState(null)
  const [pending, setPending] = useState([])
  const [modelsList, setModelsList] = useState([])
  const [busy, setBusy] = useState('')

  const load = () => {
    api.getScoreboard().then(setSb).catch(() => {})
    api.getPendingChanges().then(setPending).catch(() => {})
    api.getModels().then(setModelsList).catch(() => {})
  }
  useEffect(() => { load() }, [refreshKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const decide = async (id, approve) => {
    setBusy(id)
    try {
      if (approve) await api.approvePendingChange(id)
      else await api.rejectPendingChange(id)
    } catch (e) { /* surfaced on reload */ }
    setBusy('')
    setTimeout(load, 400)
  }

  const decideModel = async (version, promote) => {
    setBusy('m' + version)
    try {
      if (promote) await api.promoteModel(version)
      else await api.rejectModel(version)
    } catch (e) { /* surfaced on reload */ }
    setBusy('')
    setTimeout(load, 300)
  }

  const trend = sb?.trend || []
  const all = trend.flatMap((t) => [t.mean_abs_start_err, t.mean_abs_end_err]).filter((v) => v != null)
  const min = all.length ? Math.min(...all) : 0
  const max = all.length ? Math.max(...all) : 1
  const openPending = pending.filter((p) => p.status === 'pending')

  return (
    <div>
      <div className="row spread">
        <h2>Learning scoreboard — is it getting better?</h2>
        <button onClick={load}>Refresh</button>
      </div>
      <div className="muted" style={{ marginBottom: 6 }}>
        Mean absolute start/end boundary error (frames) across all labeled clips, per global
        parameter version. Lower is better — the direct answer to "is the labeling paying off?"
      </div>

      {trend.length === 0 ? (
        <div className="muted">No evaluations yet — save corrected labels to populate this.</div>
      ) : (
        <>
          <svg width="340" height="90" style={{ background: 'rgba(127,127,127,0.12)', borderRadius: 6 }}>
            <Spark vals={trend.map((t) => t.mean_abs_start_err)} color="#4cc38a" w={340} h={90} min={min} max={max} />
            <Spark vals={trend.map((t) => t.mean_abs_end_err)} color="#e0a03a" w={340} h={90} min={min} max={max} />
          </svg>
          <div className="row" style={{ gap: 14, fontSize: 12, margin: '4px 0 8px' }}>
            <span style={{ color: '#4cc38a' }}>● start error</span>
            <span style={{ color: '#e0a03a' }}>● end error</span>
            <span className="muted">(frames, lower is better)</span>
          </div>
          <table>
            <thead><tr><th>Version</th><th>Origin</th><th>Start err</th><th>End err</th><th>Match rate</th><th>Extra</th></tr></thead>
            <tbody>
              {trend.map((t) => (
                <tr key={t.version} style={t.version === sb.current_version ? { color: '#4cc38a' } : {}}>
                  <td>v{t.version}{t.version === sb.current_version ? ' (current)' : ''}</td>
                  <td className="muted">{t.origin || t.scope}</td>
                  <td>{t.mean_abs_start_err ?? '—'}f</td>
                  <td>{t.mean_abs_end_err ?? '—'}f</td>
                  <td>{t.match_rate ?? '—'}</td>
                  <td>{t.n_extra ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {openPending.length > 0 && (
        <>
          <h2>Pending changes — need your approval</h2>
          <div className="muted" style={{ marginBottom: 6 }}>
            Higher-risk Pass 1 gate changes are never auto-applied. Approving re-validates the change
            across all labeled clips (a full re-run, in the background) before committing.
          </div>
          <table>
            <thead><tr><th>Param</th><th>Change</th><th>Why</th><th /></tr></thead>
            <tbody>
              {openPending.map((p) => (
                <tr key={p.id}>
                  <td>{p.param}</td>
                  <td>{(+p.from_value).toFixed(3)} → {(+p.to_value).toFixed(3)}</td>
                  <td className="muted">{p.rationale}</td>
                  <td className="row">
                    <button className="blue" disabled={busy === p.id} onClick={() => decide(p.id, true)}>Approve</button>
                    <button disabled={busy === p.id} onClick={() => decide(p.id, false)}>Reject</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <h2>Trained models (start / end)</h2>
      <div className="muted" style={{ marginBottom: 6 }}>
        Models learn boundary detection from your accumulated labels. A new model is registered
        as <b>shadow</b> and only promoted to <b>live</b> after it beats the heuristic on held-out
        clips (needs ≥3 labeled clips). A live model is used by Pass 2 on the next run.
      </div>
      {modelsList.length === 0 ? (
        <div className="muted">
          No trained models yet. Once you have ≥3 labeled clips, run
          <code> backend/scripts/train_bounce.py</code> to train the rally-end model.
        </div>
      ) : (
        <table>
          <thead><tr><th>Version</th><th>Kind</th><th>Status</th><th>Clips</th><th>Windows</th><th>End err (held-out)</th><th /></tr></thead>
          <tbody>
            {modelsList.map((m) => (
              <tr key={m.version} style={m.status === 'live' ? { color: '#4cc38a' } : {}}>
                <td>v{m.version}</td>
                <td>{m.kind}</td>
                <td><span className={`pill ${m.status === 'live' ? 'human' : m.status === 'rejected' ? 'extra' : ''}`}>{m.status}</span></td>
                <td>{m.trained_on?.n_clips ?? '—'}</td>
                <td>{m.trained_on?.n_windows ?? '—'}</td>
                <td>{m.metrics?.after?.mean_abs_end_err ?? '—'}</td>
                <td className="row">
                  {m.status !== 'live' && <button className="blue" disabled={busy === 'm' + m.version} onClick={() => decideModel(m.version, true)}>Promote</button>}
                  {m.status !== 'rejected' && <button disabled={busy === 'm' + m.version} onClick={() => decideModel(m.version, false)}>Reject</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
