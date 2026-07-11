import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import VideoPlayer from './VideoPlayer.jsx'
import Timeline from './Timeline.jsx'
import CalibrationPanel from './CalibrationPanel.jsx'
import LabelingPanel from './LabelingPanel.jsx'

const TABS = [
  ['setup', 'Setup'],
  ['pass1', 'Pass 1 rallies'],
  ['pass2', 'Pass 2 detection'],
]

export default function Workspace({ videoId, onBack }) {
  const [video, setVideo] = useState(null)
  const [rallies, setRallies] = useState({ human: [], pass1: [], pass2: [] })
  const [humanDraft, setHumanDraft] = useState(null) // local rally-labeling draft
  const [refined, setRefined] = useState({ exists: false, rallies: [], windows: [], rejected: 0 }) // Pass 2 keep/drop verdicts
  const [tab, setTab] = useState('setup')
  const [frame, setFrame] = useState(0)
  const [calibration, setCalibration] = useState(null)
  const [courtLines, setCourtLines] = useState(null)
  const [clickMode, setClickMode] = useState(null) // {type:'calib', name, onPick}
  const [flash, setFlash] = useState('')
  const [overlayNonce, setOverlayNonce] = useState(0) // bumps overlay refetch while processing
  // Ground-truth marking mode (ball events). gt.step drives the keyboard state
  // machine: idle -> type (B/H/N after a click) -> inout (bounce confirm), or
  // edit (an existing mark was clicked).
  const [gtMode, setGtMode] = useState(false)
  const [marks, setMarks] = useState([])
  const [gt, setGt] = useState({ step: 'idle', pending: null, choice: 'in', sel: null })
  const wasActiveRef = useRef(false)
  const playerRef = useRef(null)

  const loadRallies = useCallback(() => {
    api.getRallies(videoId).then((r) => {
      setRallies(r)
      // Seed the labeling draft from saved human labels once, without clobbering
      // an in-progress edit (the 3 s job poll also calls this).
      setHumanDraft((d) => d ?? r.human.map(({ start_frame, end_frame }) => ({ start_frame, end_frame })))
    }).catch(() => {})
  }, [videoId])

  const loadRefined = useCallback(() => {
    api.getRefined(videoId).then(setRefined).catch(() => {})
  }, [videoId])

  // Grade Pass 2's keep/drop on one window; clicking the active verdict again
  // clears it. Immediate POST/DELETE then reload — get_refined re-attaches the
  // grade to the window by frame-span IoU, so the button state updates itself.
  const gradeWindow = useCallback((win, verdict) => {
    const p = (win.human_verdict === verdict && win.grade_id)
      ? api.deleteWindowGrade(win.grade_id)
      : api.addWindowGrade(videoId, {
          pass1_start_frame: win.pass1_start_frame,
          pass1_end_frame: win.pass1_end_frame,
          human_verdict: verdict,
          ai_verdict: win.ai_verdict,
          ai_rejection_reason: win.rejection_reason,
          serve_gate_score: win.serve_gate_score,
        })
    p.then(loadRefined).catch((e) => note(`grade failed: ${e.message}`))
  }, [videoId, loadRefined])

  const loadCalibration = useCallback(() => {
    api.getCalibration(videoId).then((c) => {
      setCalibration(c)
      if (c.calibrated) {
        api.projectCourt(c.id).then((p) => setCourtLines(p.lines)).catch(() => {})
      } else {
        setCourtLines(null)
      }
    }).catch(() => {})
  }, [videoId])

  useEffect(() => {
    api.getVideo(videoId).then(setVideo)
    loadRallies()
    loadRefined()
    loadCalibration()
  }, [videoId, loadRallies, loadRefined, loadCalibration])

  // Poll jobs + streamed results while processing runs in the background.
  useEffect(() => {
    const t = setInterval(() => {
      api.getVideo(videoId).then((v) => {
        setVideo(v)
        const active = v.jobs?.some((j) => j.status === 'queued' || j.status === 'running')
        // While a pass is running (and for one tick after it finishes) refetch
        // rallies + overlays: Pass 2 writes overlays per window as it goes.
        if (active || wasActiveRef.current) { loadRallies(); loadRefined(); setOverlayNonce((n) => n + 1) }
        wasActiveRef.current = active
      }).catch(() => {})
    }, 3000)
    return () => clearInterval(t)
  }, [videoId, loadRallies, loadRefined])

  const seek = (f) => playerRef.current?.seekFrame(f)
  const note = (msg) => { setFlash(msg); setTimeout(() => setFlash(''), 2500) }

  // ---- ground-truth marking -------------------------------------------------
  const loadMarks = useCallback(() => {
    api.listBallCorrections(videoId).then(setMarks).catch(() => {})
  }, [videoId])
  useEffect(() => { loadMarks() }, [loadMarks])

  const gtReset = () => setGt({ step: 'idle', pending: null, choice: 'in', sel: null })

  const saveMark = useCallback((eventType, inOut) => {
    setGt((g) => {
      if (!g.pending) return g
      // No flash note on success — the new diamond, list row, and count are
      // the confirmation; a popup on every mark shifts the layout mid-flow.
      api.addBallCorrection(videoId, {
        frame: g.pending.frame, x: g.pending.x, y: g.pending.y,
        event_type: eventType, in_out: inOut || null,
      }).then(loadMarks)
        .catch((e) => note(`save failed: ${e.message}`))
      return { step: 'idle', pending: null, choice: 'in', sel: null }
    })
  }, [videoId, loadMarks])

  const editMark = useCallback((patch) => {
    setGt((g) => {
      if (!g.sel) return g
      api.updateBallCorrection(g.sel.id, patch)
        .then(loadMarks)
        .catch((e) => note(`edit failed: ${e.message}`))
      return { ...g, step: 'idle', sel: null }
    })
  }, [loadMarks])

  const onCanvasClick = async (x, y) => {
    if (gtMode) {
      playerRef.current?.video()?.pause()
      const f = playerRef.current?.currentFrame() ?? frame
      // Click on an existing mark (same-ish frame, close in px) selects it.
      const hit = marks.find((m) => Math.abs(m.frame - f) <= 2
        && Math.hypot(m.x - x, m.y - y) < 16)
      if (gt.step === 'edit' && gt.sel && !hit) {
        // Click while a mark is selected = move it here (to the current frame).
        editMark({ x, y, frame: f })
        return
      }
      if (hit) {
        setGt({ step: 'edit', pending: null, choice: hit.in_out || 'in', sel: hit })
        return
      }
      const pending = { frame: f, x, y, proj: null }
      setGt({ step: 'type', pending, choice: 'in', sel: null })
      // In/out suggestion fetched in the background so it's ready by the time
      // B is pressed; bounces are rare enough that a race just means the
      // prompt shows "…" for a beat.
      api.projectPoint(videoId, x, y).then((p) => {
        setGt((g) => (g.pending?.frame === f && g.pending?.x === x
          ? { ...g, pending: { ...g.pending, proj: p.calibrated ? p : null },
              choice: p.calibrated ? p.suggested : 'in' }
          : g))
      }).catch(() => {})
      return
    }
    if (clickMode?.type === 'calib') clickMode.onPick?.(x, y)
  }

  // Keyboard: frame stepping + the mark state machine. Active only in marking
  // mode and never while typing in a form field.
  useEffect(() => {
    if (!gtMode) return undefined
    const onKey = (e) => {
      const t = e.target
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      const step = (d) => {
        e.preventDefault()
        const v = playerRef.current?.video()
        if (v) v.pause()
        const f = Math.max(0, (playerRef.current?.currentFrame() ?? 0) + d)
        playerRef.current?.seekFrame(f)
      }
      if (e.key === 'ArrowLeft') return step(e.shiftKey ? -10 : -1)
      if (e.key === 'ArrowRight') return step(e.shiftKey ? 10 : 1)
      const k = e.key.toLowerCase()
      if (e.key === 'Escape') { e.preventDefault(); gtReset(); return }
      if (gt.step === 'type') {
        if (k === 'b') {
          e.preventDefault()
          setGt((g) => ({ ...g, step: 'inout', choice: g.pending?.proj?.suggested || 'in' }))
        } else if (k === 'h') { e.preventDefault(); saveMark('hit') }
        else if (k === 'n') { e.preventDefault(); saveMark('position') }
      } else if (gt.step === 'inout') {
        if (e.key === 'Enter') { e.preventDefault(); saveMark('bounce', gt.choice) }
        else if (k === 'x') {
          e.preventDefault()
          setGt((g) => ({ ...g, choice: g.choice === 'in' ? 'out' : 'in' }))
        }
      } else if (gt.step === 'edit' && gt.sel) {
        if (k === 'b') { e.preventDefault(); editMark({ event_type: 'bounce' }) }
        else if (k === 'h') { e.preventDefault(); editMark({ event_type: 'hit' }) }
        else if (k === 'n') { e.preventDefault(); editMark({ event_type: 'position' }) }
        else if (k === 'x' && gt.sel.event_type === 'bounce') {
          e.preventDefault()
          editMark({ in_out: gt.sel.in_out === 'in' ? 'out' : 'in' })
        } else if (k === 'd' || e.key === 'Delete') {
          e.preventDefault()
          api.deleteBallCorrection(gt.sel.id).then(loadMarks)
          gtReset()
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [gtMode, gt, saveMark, editMark, loadMarks, videoId])

  if (!video) return <div className="muted">Loading…</div>

  const jobs = video.jobs || []
  // Newest job per (kind) that is still active — avoids showing superseded duplicates.
  const activeJobs = []
  const seenKinds = new Set()
  for (const j of jobs) {
    if ((j.status === 'queued' || j.status === 'running') && !seenKinds.has(j.kind)) {
      activeJobs.push(j)
      seenKinds.add(j.kind)
    }
  }
  activeJobs.sort((a, b) => (a.status === 'running' ? -1 : 1))
  const lastError = jobs.find((j) => j.status === 'error')

  const overlaySource = tab === 'pass1' ? 'pass1' : tab === 'pass2' ? 'pass2' : null
  const tracks = [
    { name: 'Human', color: '#4cc38a', spans: humanDraft || [] },
    { name: 'Pass 1', color: '#e5a53a', spans: rallies.pass1 },
    // Pass 2 track spans the FULL Pass 1 windows, coloured by keep/drop verdict
    // (kept = blue, dropped = grey) with a grade dot, so dropped windows are
    // visible for grading — not just the kept rallies.
    { name: 'Pass 2', color: '#58a6ff',
      spans: (refined.windows || []).map((w) => ({
        start_frame: w.pass1_start_frame, end_frame: w.pass1_end_frame,
        verdict: w.ai_verdict, human_verdict: w.human_verdict })) },
  ]

  return (
    <div>
      <div className="row spread">
        <div className="row">
          <button onClick={onBack}>← Clips</button>
          <h1 style={{ margin: 0 }}>{video.filename}</h1>
        </div>
        <div className="row">
          <span className="muted">
            {video.fps?.toFixed(2)} fps · {video.frame_count} frames · params v{video.learned_params_version ?? '—'}
          </span>
          <button className={gtMode ? 'active' : ''}
                  style={gtMode ? { borderColor: '#c3ce4a', color: '#c3ce4a' } : undefined}
                  onClick={() => { setGtMode((m) => !m); gtReset(); playerRef.current?.video()?.pause() }}
                  title="Mark ball ground truth: click the ball, then B/H/N. Arrows step frames.">
            {gtMode ? '✓ Marking ground truth' : '🎯 Mark ground truth'}
          </button>
        </div>
      </div>

      {activeJobs.length > 0 && (
        <div className="panel" style={{ padding: 10 }}>
          {activeJobs.map((jb) => (
            <div key={jb.id} className="row spread" style={{ marginBottom: 4 }}>
              <div className="row" style={{ flex: 1 }}>
                <b style={{ minWidth: 52 }}>{jb.kind}</b>
                {jb.status === 'running' ? (
                  <>
                    <progress value={jb.progress} max="1" />
                    <span className="muted">{Math.round(jb.progress * 100)}% · {jb.message}</span>
                  </>
                ) : (
                  <span className="muted">
                    queued{jb.queue_ahead != null && jb.queue_ahead > 0
                      ? ` — waiting behind ${jb.queue_ahead} job${jb.queue_ahead > 1 ? 's' : ''}`
                      : ' — next up'}
                  </span>
                )}
              </div>
              <button className="danger" onClick={() => api.cancelJob(jb.id).then(() => note(`${jb.kind} cancelled`))}>
                Cancel
              </button>
            </div>
          ))}
        </div>
      )}
      {lastError && activeJobs.length === 0 && (
        <div className="panel" style={{ padding: 8 }}>
          <span className="error">⚠ {lastError.kind} failed: {lastError.message}</span>
          {' '}<button onClick={() => api.reprocess(videoId, lastError.kind).then(() => note('re-queued'))}>Retry</button>
        </div>
      )}
      {flash && <div className="panel" style={{ padding: 8, borderColor: '#2c9e69' }}>{flash}</div>}

      <VideoPlayer
        ref={playerRef}
        videoId={videoId}
        meta={video}
        overlaySource={overlaySource}
        courtLines={courtLines}
        clickMode={gtMode ? { type: 'gt' } : clickMode}
        onCanvasClick={onCanvasClick}
        onFrame={setFrame}
        overlayNonce={overlayNonce}
        gtMarks={gtMode ? marks : null}
        pendingMark={gt.pending}
        selectedMarkId={gt.sel?.id}
      />
      <div style={{ marginTop: 8 }}>
        <Timeline frameCount={video.frame_count || 1} frame={frame} tracks={tracks}
                  marks={gtMode ? marks : null} onSeek={seek} />
      </div>

      {gtMode && (
        <GroundTruthPanel gt={gt} marks={marks} frame={frame} fps={video.fps || 30}
                          onSeek={seek}
                          onDelete={(id) => api.deleteBallCorrection(id).then(loadMarks)} />
      )}

      <div className="tabs">
        {TABS.map(([id, label]) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      <div className="tabbody">
        {tab === 'setup' && (
          <>
            <CalibrationPanel
              videoId={videoId}
              calibration={calibration}
              setClickMode={setClickMode}
              clickMode={clickMode}
              onSaved={() => {
                loadCalibration()
                // Pass 2 needs the homography — re-run it automatically now.
                api.reprocess(videoId, 'pass2')
                  .then(() => note('Calibration saved — Pass 2 re-running automatically'))
                  .catch(() => note('Calibration saved'))
              }}
            />
            <LabelingPanel
              videoId={videoId}
              fps={video.fps || 30}
              frame={frame}
              labels={humanDraft || []}
              setLabels={setHumanDraft}
              onSeek={seek}
              onSaved={() => { note('Rally labels saved'); loadRallies() }}
            />
            <ServeDatasetPanel videoId={videoId} note={note} />
          </>
        )}

        {tab === 'pass1' && (
          <PassTab source="pass1" rallies={rallies.pass1} fps={video.fps || 30} onSeek={seek} />
        )}
        {tab === 'pass2' && (
          <PassTab source="pass2" windows={refined.windows || []} exists={refined.exists}
                   error={refined.error} fps={video.fps || 30} onSeek={seek}
                   onGrade={gradeWindow} videoId={videoId} note={note} />
        )}
      </div>
    </div>
  )
}

// Turns the saved human rally labels into the model-ready serve dataset
// (rally-start = serve + ±buffer). Writes data/refine/labels/<id>.*; the ball
// ground-truth marks are left untouched.
function ServeDatasetPanel({ videoId, note }) {
  const [busy, setBusy] = useState(false)
  const [summary, setSummary] = useState(null)
  const run = async () => {
    setBusy(true)
    try {
      const r = await api.writeServeDataset(videoId)
      setSummary(r)
      note(`Serve dataset written: ${r.n_serves} serves · ${r.n_gate_negatives} negatives`)
    } catch (e) { note(`export failed: ${e.message}`) }
    setBusy(false)
  }
  return (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="row spread">
        <div>
          <b>Serve / rally dataset</b>
          <div className="muted">
            Every rally-start is a labeled serve; a ±buffer around it is held out of the
            negatives. Save the labels above first.
          </div>
        </div>
        <button className="blue" onClick={run} disabled={busy}>
          {busy ? 'Exporting…' : 'Export serve dataset'}
        </button>
      </div>
      {summary && (
        <div className="muted" style={{ marginTop: 6 }}>
          Wrote {summary.n_rallies} rallies · {summary.n_serves} serves ·{' '}
          {summary.n_gate_negatives} negatives → data/refine/labels/
        </div>
      )}
    </div>
  )
}

function PassTab({ source, rallies = [], windows = [], fps, onSeek, exists = true,
                  error, onGrade, videoId, note }) {
  if (source !== 'pass2') {
    return (
      <div>
        <div className="muted">
          Coarse gate over 100% of footage: motion energy + coarse ball + serve-formation
          spot checks. Boundaries are intentionally loose (padded + merged) to favour recall.
          Runs automatically on upload.
        </div>
        <table style={{ marginTop: 10 }}>
          <thead><tr><th>#</th><th>Start</th><th>End</th><th>Length</th><th></th></tr></thead>
          <tbody>
            {rallies.map((r, i) => (
              <tr key={r.id || i}>
                <td>{i + 1}</td>
                <td>{fmt(r.start_frame, fps)}</td>
                <td>{fmt(r.end_frame, fps)}</td>
                <td>{((r.end_frame - r.start_frame) / fps).toFixed(1)}s</td>
                <td><button onClick={() => onSeek(r.start_frame)}>▶</button></td>
              </tr>
            ))}
            {rallies.length === 0 && (
              <tr><td colSpan="5" className="muted">No rallies yet.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    )
  }

  // ---- Pass 2: keep/drop grading -------------------------------------------
  if (!exists) {
    return (
      <div className="muted">
        No Pass 2 refinement for this clip yet. Pass 2 judges each Pass 1 window
        keep-or-drop when it runs — reprocess Pass 2 (save calibration in Setup,
        or Regenerate in the Report tab) to generate the verdicts.
        {error ? <div className="error" style={{ marginTop: 4 }}>refine error: {error}</div> : null}
      </div>
    )
  }

  // Live keep/drop accuracy, derived from the graded windows in view.
  const graded = windows.filter((w) => w.human_verdict)
  let nCorrect = 0, droppedWrong = 0, keptWrong = 0
  const byReason = {}
  for (const w of graded) {
    const aiKeep = w.ai_verdict === 'rally'
    const humanKeep = w.human_verdict === 'keep'
    if (aiKeep === humanKeep) nCorrect++
    else if (aiKeep) keptWrong++
    else { droppedWrong++; const r = w.rejection_reason || 'unknown'; byReason[r] = (byReason[r] || 0) + 1 }
  }
  const acc = graded.length ? Math.round((100 * nCorrect) / graded.length) : null
  const reasonStr = Object.entries(byReason).map(([k, v]) => `${k} ${v}`).join(', ')

  return (
    <div>
      <div className="muted">
        Each Pass 1 window with Pass 2’s keep/drop verdict. Grade 👍 (real rally,
        should keep) or 👎 (not a rally, should drop) — the score below updates
        live and drives the serve-gate recommendation. Keep/drop only; boundaries
        are out of scope here.
      </div>
      <div className="row spread" style={{ marginTop: 8, alignItems: 'baseline' }}>
        <div>
          {graded.length ? (
            <b>Keep/drop accuracy {acc}% ({nCorrect}/{graded.length})</b>
          ) : <b className="muted">No windows graded yet</b>}
          {graded.length > 0 && (
            <span className="muted">
              {' · '}dropped-wrong {droppedWrong}{reasonStr ? ` (${reasonStr})` : ''}
              {' · '}kept-wrong {keptWrong}
            </span>
          )}
        </div>
      </div>
      <Recommendation videoId={videoId} note={note} nGraded={graded.length} />
      <table style={{ marginTop: 10 }}>
        <thead><tr>
          <th>#</th><th>Start</th><th>End</th><th>Length</th>
          <th>AI verdict</th><th>gate</th><th>Grade</th><th></th>
        </tr></thead>
        <tbody>
          {windows.map((w, i) => {
            const kept = w.ai_verdict === 'rally'
            const wrong = w.human_verdict && (kept !== (w.human_verdict === 'keep'))
            return (
              <tr key={i} style={wrong ? { background: '#3a2530' } : undefined}>
                <td>{i + 1}</td>
                <td>{fmt(w.pass1_start_frame, fps)}</td>
                <td>{fmt(w.pass1_end_frame, fps)}</td>
                <td>{((w.pass1_end_frame - w.pass1_start_frame) / fps).toFixed(1)}s</td>
                <td style={{ color: kept ? '#58a6ff' : '#b08' }}>
                  {kept ? 'KEPT' : `dropped: ${w.rejection_reason || '—'}`}
                  {w.is_split ? ' (split)' : ''}
                </td>
                <td className="muted">{w.serve_gate_score != null ? w.serve_gate_score.toFixed(2) : '—'}</td>
                <td>
                  <button className={w.human_verdict === 'keep' ? 'active' : ''}
                          style={w.human_verdict === 'keep' ? { borderColor: '#4cc38a', color: '#4cc38a' } : undefined}
                          title="Real rally — should keep" onClick={() => onGrade(w, 'keep')}>👍</button>
                  {' '}
                  <button className={w.human_verdict === 'drop' ? 'active' : ''}
                          style={w.human_verdict === 'drop' ? { borderColor: '#ff5470', color: '#ff5470' } : undefined}
                          title="Not a rally — should drop" onClick={() => onGrade(w, 'drop')}>👎</button>
                </td>
                <td><button onClick={() => onSeek(w.pass1_start_frame)}>▶</button></td>
              </tr>
            )
          })}
          {windows.length === 0 && (
            <tr><td colSpan="8" className="muted">Pass 2 produced no windows here.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

// Read-only serve-gate recommendation from the accumulated grades. Applying
// writes the override YAML; Pass 2 must be reprocessed for it to take effect.
function Recommendation({ videoId, note, nGraded }) {
  const [rec, setRec] = useState(null)
  const [busy, setBusy] = useState(false)
  const load = async () => {
    setBusy(true)
    try { setRec(await api.getGradeRecommendation(videoId)) }
    catch (e) { note(`recommendation failed: ${e.message}`) }
    setBusy(false)
  }
  const apply = async (min) => {
    try {
      await api.writeRefineOverride({ serve_gate_min: min })
      note(`serve_gate_min = ${min} written — reprocess Pass 2 to apply`)
    } catch (e) { note(`apply failed: ${e.message}`) }
  }
  return (
    <div className="panel" style={{ padding: 8, marginTop: 8 }}>
      <div className="row spread">
        <div className="muted">
          Recommend the serve-gate threshold that best matches your grades (pools
          grades across all clips).
        </div>
        <button className="blue" onClick={load} disabled={busy}>
          {busy ? 'Sweeping…' : 'Recommend better serve gate'}
        </button>
      </div>
      {rec && !rec.ready && <div className="muted" style={{ marginTop: 6 }}>{rec.reason}</div>}
      {rec && rec.ready && (
        <div style={{ marginTop: 6 }}>
          <div>
            <b>serve_gate_min {rec.current.serve_gate_min} → {rec.recommended.overrides.serve_gate_min}</b>
            {' '}<span className="muted">
              accuracy {(rec.current.accuracy * 100).toFixed(0)}% → {(rec.recommended.accuracy * 100).toFixed(0)}%
              {' '}({rec.recommended.delta >= 0 ? '+' : ''}{(rec.recommended.delta * 100).toFixed(0)} pts)
              {' · '}rescues {rec.current.dropped_wrong - rec.recommended.dropped_wrong} wrong-drops
              {' · '}n={rec.n_graded}
            </span>
          </div>
          {rec.recommended.overrides.serve_gate_min !== rec.current.serve_gate_min ? (
            <button style={{ marginTop: 6 }}
                    onClick={() => apply(rec.recommended.overrides.serve_gate_min)}>
              Apply serve_gate_min = {rec.recommended.overrides.serve_gate_min} (writes config; re-run Pass 2)
            </button>
          ) : <div className="muted" style={{ marginTop: 6 }}>Current threshold is already best on your grades.</div>}
        </div>
      )}
    </div>
  )
}

// Prompt bar + mark list for ground-truth marking mode. The prompt mirrors the
// keyboard state machine in Workspace so the next keypress is always spelled
// out on screen.
function GroundTruthPanel({ gt, marks, frame, fps, onSeek, onDelete }) {
  let prompt
  if (gt.step === 'type') {
    prompt = <b>B = bounce · H = hit · N = ball position only · Esc = cancel</b>
  } else if (gt.step === 'inout') {
    const p = gt.pending?.proj
    prompt = (
      <b>
        Bounce — homography says {p ? p.suggested.toUpperCase() : '…'}
        {p ? ` (${p.line_dist_m}m from nearest line)` : ''} → saving as{' '}
        <span style={{ color: gt.choice === 'in' ? '#00e5ff' : '#ff5470' }}>
          {gt.choice.toUpperCase()}
        </span>
        {' '}· Enter = save · X = flip · Esc = cancel
      </b>
    )
  } else if (gt.step === 'edit' && gt.sel) {
    prompt = (
      <b>
        Editing {gt.sel.event_type} @ f{gt.sel.frame} — B/H/N = retype
        {gt.sel.event_type === 'bounce' ? ' · X = flip in/out' : ''} · click = move here ·
        D = delete · Esc = done
      </b>
    )
  } else {
    prompt = <span>Click the ball to mark it · ←/→ = ±1 frame · Shift+←/→ = ±10 · click a diamond to edit</span>
  }
  const disagreements = marks.filter((m) => m.disagree).length
  return (
    <div className="panel" style={{ padding: 10, borderColor: '#c3ce4a' }}>
      <div className="row spread">
        <div>{prompt}</div>
        <div className="muted">
          {marks.length} mark{marks.length === 1 ? '' : 's'} on this clip
          {disagreements > 0 && ` · ${disagreements} in/out disagreement${disagreements === 1 ? '' : 's'} ⚠`}
        </div>
      </div>
      {marks.length > 0 && (
        <table style={{ marginTop: 8, width: '100%' }}>
          <thead>
            <tr><th>Frame</th><th>Type</th><th>In/out</th><th>Court (m)</th><th></th><th></th></tr>
          </thead>
          <tbody>
            {marks.map((m) => (
              <tr key={m.id} style={m.frame === frame ? { background: '#1c2530' } : undefined}>
                <td>{fmt(m.frame, fps)}</td>
                <td>{m.event_type}</td>
                <td>
                  {m.event_type === 'bounce'
                    ? <>{m.in_out}{m.disagree ? <span title={`homography suggested ${m.in_out_suggested}`}> ⚠</span> : ''}</>
                    : '—'}
                </td>
                <td>{m.court_x != null ? `${m.court_x.toFixed(2)}, ${m.court_y.toFixed(2)}` : '—'}</td>
                <td><button onClick={() => onSeek(m.frame)}>▶</button></td>
                <td><button className="danger" onClick={() => onDelete(m.id)}>✕</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function fmt(f, fps) {
  const s = f / fps
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')} (${f})`
}
