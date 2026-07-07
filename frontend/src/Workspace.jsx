import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import VideoPlayer from './VideoPlayer.jsx'
import Timeline from './Timeline.jsx'
import CalibrationPanel from './CalibrationPanel.jsx'

const TABS = [
  ['setup', 'Setup'],
  ['pass1', 'Pass 1 rallies'],
  ['pass2', 'Pass 2 detection'],
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
  const wasActiveRef = useRef(false)
  const playerRef = useRef(null)

  const loadRallies = useCallback(() => {
    api.getRallies(videoId).then(setRallies).catch(() => {})
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

  const onCanvasClick = async (x, y) => {
    if (clickMode?.type === 'calib') clickMode.onPick?.(x, y)
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

  const overlaySource = tab === 'pass1' ? 'pass1' : tab === 'pass2' ? 'pass2' : null
  const tracks = [
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
        overlayNonce={overlayNonce}
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

function fmt(f, fps) {
  const s = f / fps
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')} (${f})`
}
