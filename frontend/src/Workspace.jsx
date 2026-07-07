import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import VideoPlayer from './VideoPlayer.jsx'
import Timeline from './Timeline.jsx'
import LabelingPanel from './LabelingPanel.jsx'
import CalibrationPanel from './CalibrationPanel.jsx'
import NotesPanel from './NotesPanel.jsx'
import ReportPanel from './ReportPanel.jsx'
import ChatPanel from './ChatPanel.jsx'

const TABS = [
  ['label', 'Label rallies'],
  ['pass1', 'Pass 1 results'],
  ['pass2', 'Pass 2 results'],
  ['compare', 'Compare'],
  ['report', 'Report'],
  ['chat', '💬 Assistant'],
]

export default function Workspace({ videoId, onBack }) {
  const [video, setVideo] = useState(null)
  const [rallies, setRallies] = useState({ human: [], pass1: [], pass2: [] })
  const [humanDraft, setHumanDraft] = useState(null) // local labeling state
  const [tab, setTab] = useState('label')
  const [frame, setFrame] = useState(0)
  const [calibration, setCalibration] = useState(null)
  const [courtLines, setCourtLines] = useState(null)
  const [clickMode, setClickMode] = useState(null) // {type:'calib', name} | {type:'ball'}
  const [flash, setFlash] = useState('')
  const playerRef = useRef(null)

  const loadRallies = useCallback(() => {
    api.getRallies(videoId).then((r) => {
      setRallies(r)
      setHumanDraft((d) => d ?? r.human.map(({ start_frame, end_frame }) => ({ start_frame, end_frame })))
    }).catch(() => {})
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
        if (active) loadRallies()
      }).catch(() => {})
    }, 3000)
    return () => clearInterval(t)
  }, [videoId, loadRallies])

  const seek = (f) => playerRef.current?.seekFrame(f)
  const note = (msg) => { setFlash(msg); setTimeout(() => setFlash(''), 2500) }

  const onCanvasClick = async (x, y) => {
    if (clickMode?.type === 'ball') {
      await api.addBallCorrection(videoId, { frame, x, y })
      note(`Ball spot-correction saved at frame ${frame} (clip-local; applied on next Pass 2 run)`)
      setClickMode(null)
    } else if (clickMode?.type === 'calib') {
      clickMode.onPick?.(x, y)
    }
  }

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

  const overlaySource = tab === 'pass1' ? 'pass1' : (tab === 'pass2' || tab === 'compare') ? 'pass2' : null
  const tracks = [
    { name: 'Human', color: '#4cc38a', spans: humanDraft || [] },
    { name: 'Pass 1', color: '#e5a53a', spans: rallies.pass1 },
    { name: 'Pass 2', color: '#58a6ff', spans: rallies.pass2 },
  ]

  return (
    <div>
      <div className="row spread">
        <div className="row">
          <button onClick={onBack}>← Clips</button>
          <h1 style={{ margin: 0 }}>{video.filename}</h1>
        </div>
        <div className="muted">
          {video.fps?.toFixed(2)} fps · {video.frame_count} frames · params v{video.learned_params_version ?? '—'}
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
        clickMode={clickMode}
        onCanvasClick={onCanvasClick}
        onFrame={setFrame}
      />
      <div style={{ marginTop: 8 }}>
        <Timeline frameCount={video.frame_count || 1} frame={frame} tracks={tracks} onSeek={seek} />
      </div>

      <div className="tabs">
        {TABS.map(([id, label]) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      <div className="tabbody">
        {tab === 'label' && humanDraft && (
          <>
            <LabelingPanel
              videoId={videoId}
              fps={video.fps || 30}
              frame={frame}
              labels={humanDraft}
              setLabels={setHumanDraft}
              onSeek={seek}
              onSaved={() => { note('Labels saved'); loadRallies() }}
            />
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
            <div className="panel">
              <h2>Optional: ball spot-correction</h2>
              <div className="row">
                <button
                  className={clickMode?.type === 'ball' ? 'blue' : ''}
                  onClick={() => setClickMode(clickMode?.type === 'ball' ? null : { type: 'ball' })}>
                  {clickMode?.type === 'ball' ? 'Click the ball in the video… (esc: click again)' : 'Mark ball at current frame'}
                </button>
                <span className="muted">Rarely needed. Clip-local — never touches the global baseline.</span>
              </div>
            </div>
            <NotesPanel videoId={videoId} videoTime={frame / (video.fps || 30)} />
          </>
        )}

        {(tab === 'pass1' || tab === 'pass2') && (
          <PassTab source={tab} rallies={rallies[tab]} fps={video.fps || 30} onSeek={seek} />
        )}

        {tab === 'compare' && (
          <CompareTab human={humanDraft || []} pass2={rallies.pass2} pass1={rallies.pass1} fps={video.fps || 30} onSeek={seek} />
        )}

        {tab === 'report' && (
          <>
            <ReportPanel videoId={videoId} onRegenerate={(which) =>
              api.reprocess(videoId, which).then(() => note(`${which} re-queued — overlays regenerate from metadata`))} />
            <NotesPanel videoId={videoId} videoTime={frame / (video.fps || 30)} />
          </>
        )}

        {tab === 'chat' && (
          <ChatPanel
            videoId={videoId}
            fps={video.fps || 30}
            currentFrame={frame}
            onStateChange={() => { loadRallies(); loadCalibration(); api.getVideo(videoId).then(setVideo) }}
          />
        )}
      </div>
    </div>
  )
}

function PassTab({ source, rallies, fps, onSeek }) {
  return (
    <div>
      <div className="muted">
        {source === 'pass1'
          ? 'Coarse gate over 100% of footage: motion energy + coarse ball + serve-formation spot checks. Boundaries are intentionally rough. Runs automatically on upload.'
          : 'Fine boundaries from full-rate ball tracking, height reconstruction, and terminal-event logic inside padded Pass 1 windows. Runs automatically after Pass 1 (and after calibration). Re-runs live in the Report tab.'}
      </div>
      <table style={{ marginTop: 10 }}>
        <thead><tr><th>#</th><th>Start</th><th>End</th><th>Length</th><th>{source === 'pass2' ? 'End reason' : ''}</th><th></th></tr></thead>
        <tbody>
          {rallies.map((r, i) => (
            <tr key={r.id || i}>
              <td>{i + 1}</td>
              <td>{fmt(r.start_frame, fps)}</td>
              <td>{fmt(r.end_frame, fps)}</td>
              <td>{((r.end_frame - r.start_frame) / fps).toFixed(1)}s</td>
              <td className="muted">
                {r.meta?.end_event ? `${r.meta.end_event.why}${r.meta.confirmed ? ' ✓confirmed' : ''}` : r.meta?.end_source || ''}
              </td>
              <td><button onClick={() => onSeek(r.start_frame)}>▶</button></td>
            </tr>
          ))}
          {rallies.length === 0 && <tr><td colSpan="6" className="muted">No rallies yet.</td></tr>}
        </tbody>
      </table>
    </div>
  )
}

function iou(a, b) {
  const inter = Math.max(0, Math.min(a.end_frame, b.end_frame) - Math.max(a.start_frame, b.start_frame))
  const union = Math.max(a.end_frame, b.end_frame) - Math.min(a.start_frame, b.start_frame)
  return union > 0 ? inter / union : 0
}

function CompareTab({ human, pass2, pass1, fps, onSeek }) {
  const rows = human.map((h, i) => {
    let best = null, bestIou = 0
    for (const a of pass2) {
      const v = iou(h, a)
      if (v > bestIou) { bestIou = v; best = a }
    }
    return { i, h, a: bestIou > 0.1 ? best : null, iou: bestIou }
  })
  const matchedIds = new Set(rows.filter((r) => r.a).map((r) => r.a.id))
  const extras = pass2.filter((a) => !matchedIds.has(a.id))
  return (
    <div>
      <div className="muted">
        Your labels vs AI detections over the same timeline (tracks above the tabs).
        Generate the report for full metrics and the LLM analysis.
      </div>
      <table style={{ marginTop: 10 }}>
        <thead><tr><th>Human rally</th><th>Pass 2 match</th><th>IoU</th><th>Start Δ</th><th>End Δ</th><th></th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.i}>
              <td>{fmt(r.h.start_frame, fps)} → {fmt(r.h.end_frame, fps)}</td>
              <td>{r.a ? `${fmt(r.a.start_frame, fps)} → ${fmt(r.a.end_frame, fps)}` : <span className="pill missed">missed</span>}</td>
              <td>{r.a ? r.iou.toFixed(2) : '—'}</td>
              <td>{r.a ? `${r.a.start_frame - r.h.start_frame}f` : '—'}</td>
              <td>{r.a ? `${r.a.end_frame - r.h.end_frame}f` : '—'}</td>
              <td><button onClick={() => onSeek(r.h.start_frame)}>▶</button></td>
            </tr>
          ))}
          {extras.map((a) => (
            <tr key={a.id}>
              <td><span className="pill extra">extra</span></td>
              <td>{fmt(a.start_frame, fps)} → {fmt(a.end_frame, fps)}</td>
              <td>—</td><td>—</td><td>—</td>
              <td><button onClick={() => onSeek(a.start_frame)}>▶</button></td>
            </tr>
          ))}
          {rows.length === 0 && <tr><td colSpan="6" className="muted">Label rallies first.</td></tr>}
        </tbody>
      </table>
      {pass1.length > 0 && pass2.length === 0 && (
        <div className="muted" style={{ marginTop: 8 }}>
          Pass 2 has no results yet — it may still be running, or the court needs calibration.
        </div>
      )}
    </div>
  )
}

function fmt(f, fps) {
  const s = f / fps
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')} (${f})`
}
