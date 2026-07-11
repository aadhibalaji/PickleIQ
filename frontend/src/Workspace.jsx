import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import VideoPlayer from './VideoPlayer.jsx'
import Timeline from './Timeline.jsx'
import CalibrationPanel from './CalibrationPanel.jsx'

const TABS = [
  ['setup', 'Setup'],
  ['pass1', 'Pass 1 rallies'],
  ['pass2', 'Pass 2 detection'],
  ['balleval', 'Ball Eval'],
]

export default function Workspace({ videoId, onBack }) {
  const [video, setVideo] = useState(null)
  const [rallies, setRallies] = useState({ human: [], pass1: [], pass2: [] })
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
  const [serves, setServes] = useState([])   // AI-inferred serve frames (pass2 overlay)
  const [gt, setGt] = useState({ step: 'idle', pending: null, choice: 'in', sel: null })
  const wasActiveRef = useRef(false)
  const playerRef = useRef(null)

  const loadRallies = useCallback(() => {
    api.getRallies(videoId).then(setRallies).catch(() => {})
    api.getServes(videoId).then((r) => setServes(r.serves || [])).catch(() => {})
  }, [videoId])

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
    loadCalibration()
  }, [videoId, loadRallies, loadCalibration])

  // Poll jobs + streamed results while processing runs in the background.
  useEffect(() => {
    const t = setInterval(() => {
      api.getVideo(videoId).then((v) => {
        setVideo(v)
        const active = v.jobs?.some((j) => j.status === 'queued' || j.status === 'running')
        // While a pass is running (and for one tick after it finishes) refetch
        // rallies + overlays: Pass 2 writes overlays per window as it goes.
        if (active || wasActiveRef.current) { loadRallies(); setOverlayNonce((n) => n + 1) }
        wasActiveRef.current = active
      }).catch(() => {})
    }, 3000)
    return () => clearInterval(t)
  }, [videoId, loadRallies])

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

  const overlaySource =
    tab === 'pass1' ? 'pass1'
    : tab === 'pass2' ? 'pass2'
    : calibration?.calibrated ? 'pass2'
    : null
  const tracks = [
    { name: 'Pass 1', color: '#e5a53a', spans: rallies.pass1 },
    { name: 'Pass 2', color: '#58a6ff', spans: rallies.pass1 },
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
          {/* Re-run in place. Each pass wipes only its own rally_labels/overlays,
              so ground-truth marks and human labels survive — no re-upload. */}
          <button disabled={activeJobs.length > 0}
                  onClick={() => api.reprocess(videoId, 'pass1').then(() => note('Pass 1 re-running (Pass 2 chains after)'))}
                  title="Re-run Pass 1, then Pass 2. Keeps ground-truth marks and human labels.">
            ↻ Pass 1
          </button>
          <button disabled={activeJobs.length > 0}
                  onClick={() => api.reprocess(videoId, 'pass2').then(() => note('Pass 2 re-running'))}
                  title="Re-run Pass 2 only. Keeps ground-truth marks and human labels.">
            ↻ Pass 2
          </button>
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
                  marks={gtMode ? marks : null} serves={serves} onSeek={seek} />
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
        )}

        {(tab === 'pass1' || tab === 'pass2') && (
          <PassTab source={tab} rallies={rallies[tab]} fps={video.fps || 30} onSeek={seek} />
        )}

        {tab === 'balleval' && (
          <BallEvalTab videoId={videoId} jobs={jobs} fps={video.fps || 30} onSeek={seek} />
        )}
      </div>
    </div>
  )
}

function PassTab({ source, rallies, fps, onSeek }) {
  if (source === 'pass2') {
    return (
      <div className="muted">
        Pass 2 no longer produces its own rally boundaries. It runs full-rate
        player and ball detection inside each Pass 1 rally window and writes the
        accurate overlay you see on the video above (players, ball, height,
        bounce/hit markers). Rally start/end come from <b>Pass 1</b> — see the
        Pass 1 rallies tab. Re-runs automatically after you save calibration in the Setup tab.
      </div>
    )
  }
  return (
    <div>
      <div className="muted">
        Coarse gate over 100% of footage: motion energy + coarse ball + serve-formation
        spot checks. Now the sole source of rally start/end — boundaries are intentionally
        loose (padded + merged) to favour recall. Runs automatically on upload.
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
          {rallies.length === 0 && <tr><td colSpan="5" className="muted">No rallies yet.</td></tr>}
        </tbody>
      </table>
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

// ---- Ball Eval -------------------------------------------------------------
// Surfaces scripts/eval_ball_filter.py in the UI: run the harness as a
// background job, then read back the pooled base-vs-filtered comparison, the
// failure attribution, the condition slices, and the failure-crop gallery.

const pctS = (n, d) => (d ? `${((100 * n) / d).toFixed(1)}%` : 'n/a')

const median = (a) => {
  if (!a || !a.length) return null
  const s = [...a].sort((x, y) => x - y)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

function BallEvalTab({ videoId, jobs, fps, onSeek }) {
  const [runs, setRuns] = useState([])
  const [selId, setSelId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [err, setErr] = useState('')
  const [detector, setDetector] = useState('classical')
  const [diagnose, setDiagnose] = useState(true)
  const [tag, setTag] = useState('run')
  const wasRunning = useRef(false)

  const job = jobs.find((jb) => jb.kind === 'ball_eval'
    && (jb.status === 'queued' || jb.status === 'running'))

  const load = useCallback(
    () => api.listBallEvals(videoId).then(setRuns).catch((e) => setErr(String(e.message || e))),
    [videoId])
  useEffect(() => { load() }, [load])

  // The clip-level poll in Workspace refreshes `jobs`; when the eval job stops
  // being active, the new row exists — pull the list again.
  useEffect(() => {
    if (wasRunning.current && !job) load()
    wasRunning.current = !!job
  }, [job, load])

  useEffect(() => { if (!selId && runs.length) setSelId(runs[0].id) }, [runs, selId])

  useEffect(() => {
    if (!selId) { setDetail(null); return }
    let stale = false
    api.getBallEval(videoId, selId)
      .then((d) => { if (!stale) setDetail(d) })
      .catch((e) => setErr(String(e.message || e)))
    return () => { stale = true }
  }, [videoId, selId])

  const start = () => {
    setErr('')
    api.runBallEval(videoId, { detector, diagnose, tag })
      .catch((e) => setErr(String(e.message || e)))
  }

  return (
    <div>
      <div className="muted">
        Runs the ball-filter harness over this clip's rally windows: the pre-filter
        Pass 2 path (<b>base</b>) against the Phase-1 filtering layer (<b>filt</b>).
        Metrics are scored against the ground-truth marks you place with 🎯 Mark
        ground truth. Read-only — nothing here changes detection.
      </div>

      <div className="row" style={{ marginTop: 10 }}>
        <button onClick={start} disabled={!!job}>
          {job ? `Running… ${Math.round((job.progress || 0) * 100)}%` : 'Run eval'}
        </button>
        <label className="muted">detector{' '}
          <select value={detector} onChange={(e) => setDetector(e.target.value)} disabled={!!job}>
            <option value="classical">classical</option>
            <option value="tracknet">tracknet</option>
          </select>
        </label>
        <label className="muted">
          <input type="checkbox" checked={diagnose} disabled={!!job}
                 onChange={(e) => setDiagnose(e.target.checked)} /> diagnose
        </label>
        <label className="muted">tag{' '}
          <input value={tag} size="10" disabled={!!job}
                 onChange={(e) => setTag(e.target.value)} />
        </label>
        {job && <span className="muted">{job.message}</span>}
      </div>
      {err && <div className="error" style={{ marginTop: 6 }}>⚠ {err}</div>}

      <h3>Past runs</h3>
      <table>
        <thead>
          <tr><th>When</th><th>Tag</th><th>Detector</th><th>Marks</th><th>Det rate</th>
            <th>Pos err</th><th>Bounce P/R</th><th>In/out</th><th></th></tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} style={r.id === selId ? { background: '#1c2530' } : undefined}>
              <td>{new Date(r.created_at * 1000).toLocaleString()}</td>
              <td>{r.tag}</td>
              <td>{r.detector}</td>
              <td>{r.status === 'skipped' ? '—' : r.n_marks}</td>
              <td>{r.det_rate != null ? `${(r.det_rate * 100).toFixed(1)}%` : '—'}</td>
              <td>{r.pos_err_med_px != null ? `${r.pos_err_med_px.toFixed(1)}px` : '—'}</td>
              <td>
                {r.bounce_p != null && r.bounce_r != null
                  ? `${(r.bounce_p * 100).toFixed(0)}% / ${(r.bounce_r * 100).toFixed(0)}%`
                  : '—'}
              </td>
              <td>{r.inout_acc != null ? `${(r.inout_acc * 100).toFixed(0)}%` : '—'}</td>
              <td><button onClick={() => setSelId(r.id)}>View</button></td>
            </tr>
          ))}
          {runs.length === 0 && (
            <tr><td colSpan="9" className="muted">No eval runs yet — hit “Run eval”.</td></tr>
          )}
        </tbody>
      </table>

      {detail && detail.status === 'skipped' && (
        <div className="error" style={{ marginTop: 10 }}>
          ⚠ This clip could not be evaluated: {detail.message}
        </div>
      )}
      {detail && detail.status === 'ok' && detail.result && (
        <EvalDetail detail={detail} fps={fps} onSeek={onSeek} />
      )}
    </div>
  )
}

function EvalDetail({ detail, fps, onSeek }) {
  const m = detail.result
  const fr = m.frames || 1
  const b = m.base
  const fl = m.filt
  const gt = m.gt || {}
  const diag = m.diag || null

  // Mirrors fmt_clip()'s base/filt comparison, in the same order.
  const rows = [
    ['det rate', pctS(b.det, fr),
      `${pctS(fl.det, fr)} (+interp → ${pctS(fl.det + fl.interp, fr)})`],
    ['phys viol %', `${pctS(b.viol, b.pairs)} of ${b.pairs}`,
      `${pctS(fl.viol, fl.pairs)} of ${fl.pairs}`],
    ['foot-zone FP %', pctS(b.foot, b.det), pctS(fl.foot, fl.det + fl.interp)],
    ['pos err median px', fmtErr(b), fmtErr(fl)],
  ]
  if (gt.bounce) {
    rows.push(['bounce P / R', fmtPR(b, gt.bounce), fmtPR(fl, gt.bounce)])
    rows.push(['in/out accuracy', fmtAcc(b), fmtAcc(fl)])
  }

  return (
    <div>
      <h3>Pooled metrics</h3>
      <div className="muted" style={{ marginBottom: 6 }}>
        {m.windows} {m.win_source} rallies · {fr} frames · {m.raw_cands} raw candidates ·{' '}
        {gt.in_windows || 0} GT marks in windows ({gt.bounce || 0} bounce, {gt.hit || 0} hit,{' '}
        {gt.position || 0} position) · params v{detail.params_version || '—'} · overlay cover{' '}
        <span style={m.overlay_cov / fr < 0.5 ? { color: 'var(--danger)', fontWeight: 600 } : undefined}>
          {pctS(m.overlay_cov, fr)}
        </span>
        {m.overlay_cov / fr < 0.5 && ' — run Pass 2 first, the filter has no player boxes to work with'}
      </div>
      <table>
        <thead><tr><th>Metric</th><th>Base (no filter)</th><th>Filtered</th></tr></thead>
        <tbody>
          {rows.map(([k, bv, fv]) => (
            <tr key={k}><td>{k}</td><td>{bv}</td><td>{fv}</td></tr>
          ))}
        </tbody>
      </table>

      {!diag && (
        <div className="muted" style={{ marginTop: 10 }}>
          This run was made without <b>diagnose</b>, so there is no failure
          attribution, slice table, or gallery. Re-run with it checked.
        </div>
      )}

      {diag && (
        <>
          <h3>Failure attribution</h3>
          <div className="muted" style={{ marginBottom: 6 }}>
            Every GT mark the filtered pipeline missed, attributed to the pipeline
            stage that lost the ball.
          </div>
          <table>
            <thead><tr><th>Category</th><th>Count</th><th>Suggested lever</th></tr></thead>
            <tbody>
              {detail.attribution.map((a) => (
                <tr key={a.category}>
                  <td>{a.category}</td><td>{a.count}</td>
                  <td className="muted">{a.lever}</td>
                </tr>
              ))}
              {detail.attribution.length === 0 && (
                <tr><td colSpan="3" className="muted">No mark failures.</td></tr>
              )}
            </tbody>
          </table>

          {detail.bounce_attribution.length > 0 && (
            <>
              <h3>Bounce-event misses</h3>
              <table>
                <thead><tr><th>Category</th><th>Count</th><th>Suggested lever</th></tr></thead>
                <tbody>
                  {detail.bounce_attribution.map((a) => (
                    <tr key={a.category}>
                      <td>{a.category}</td><td>{a.count}</td>
                      <td className="muted">{a.lever}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          <h3>Condition slices</h3>
          <div className="muted" style={{ marginBottom: 6 }}>
            On-ball = final output within 25px of the mark. Pooled median{' '}
            {diag.pooled_med_px?.toFixed(1)}px. <span style={{ color: 'var(--danger)' }}>*HIGH</span>{' '}
            = median error over 2× pooled; <i>~lowN</i> = fewer than 8 marks, treat as noise.
          </div>
          <table>
            <thead>
              <tr><th>Dim</th><th>Value</th><th>n</th><th>On-ball %</th><th>Median err</th><th>Flags</th></tr>
            </thead>
            <tbody>
              {diag.slices.map(([dim, val, n, onball, med, flag], i) => {
                const high = flag.includes('*HIGH')
                return (
                  <tr key={`${dim}-${val}-${i}`}>
                    <td>{dim}</td>
                    <td>{val}</td>
                    <td>{n}</td>
                    <td>{onball.toFixed(1)}%</td>
                    <td style={high ? { color: 'var(--danger)', fontWeight: 600 } : undefined}>
                      {med != null ? `${med.toFixed(1)}px` : 'n/a'}
                    </td>
                    <td>
                      {high && <span style={{ color: 'var(--danger)' }}>*HIGH </span>}
                      {flag.includes('~lowN') && <span className="muted">~lowN</span>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          <Gallery gallery={diag.gallery || []} fps={fps} onSeek={onSeek} />
        </>
      )}
    </div>
  )
}

// Failure crops grouped by category. Yellow = GT mark, white = raw candidate,
// red = filter-rejected, green = final output (see _save_galleries).
function Gallery({ gallery, fps, onSeek }) {
  if (!gallery.length) {
    return (
      <>
        <h3>Failure gallery</h3>
        <div className="muted">No crops written for this run.</div>
      </>
    )
  }
  const byCat = {}
  for (const g of gallery) (byCat[g.category] ||= []).push(g)
  return (
    <>
      <h3>Failure gallery</h3>
      <div className="muted" style={{ marginBottom: 6 }}>
        Up to 10 worst crops per category. Yellow = ground truth · white = raw
        candidate · red = rejected by the filter · green = final output. Click a
        crop to seek there.
      </div>
      {Object.entries(byCat).map(([cat, items]) => (
        <div key={cat} style={{ marginBottom: 12 }}>
          <div style={{ marginBottom: 4 }}><b>{cat}</b> <span className="muted">({items.length})</span></div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {items.map((g) => (
              <figure key={g.file} style={{ margin: 0, cursor: 'pointer' }}
                      onClick={() => onSeek(g.frame)} title={`seek to frame ${g.frame}`}>
                <img src={api.evalDiagUrl(g.file)} alt={`${cat} frame ${g.frame}`}
                     loading="lazy" width="150"
                     style={{ display: 'block', borderRadius: 4, border: '1px solid var(--border)' }} />
                <figcaption className="muted" style={{ textAlign: 'center' }}>
                  {fmt(g.frame, fps)}
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      ))}
    </>
  )
}

function fmtErr(d) {
  const med = median(d.pos_err)
  return med == null
    ? `n/a (miss ${d.pos_miss})`
    : `${med.toFixed(1)}px (n=${d.pos_err.length}, miss ${d.pos_miss})`
}

function fmtPR(d, nGtBounce) {
  const p = d.bounce_pred ? (100 * d.bounce_matched) / d.bounce_pred : 0
  const r = (100 * d.bounce_matched) / nGtBounce
  return `P ${p.toFixed(0)}% R ${r.toFixed(0)}% (${d.bounce_matched}/${d.bounce_pred} pred)`
}

function fmtAcc(d) {
  return d.inout_n ? `${((100 * d.inout_ok) / d.inout_n).toFixed(0)}% of ${d.inout_n}` : 'n/a'
}

function fmt(f, fps) {
  const s = f / fps
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')} (${f})`
}
