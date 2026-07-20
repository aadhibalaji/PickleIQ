import React, { useEffect, useState } from 'react'
import { api } from './api.js'

/*
One-time-per-camera-setup court calibration. Because the camera angle is
fixed, the saved homography is GLOBAL by default and reused for every future
clip; mark it clip-local only when this specific clip's camera was bumped.
*/
export default function CalibrationPanel({ videoId, calibration, clickMode, setClickMode, onSaved }) {
  const [refPoints, setRefPoints] = useState([])
  const [picked, setPicked] = useState({}) // name -> {x, y}
  const [scope, setScope] = useState('global')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => { api.referencePoints().then(setRefPoints).catch(() => {}) }, [])
  useEffect(() => {
    if (calibration?.calibrated && calibration.corner_points) {
      const m = {}
      for (const p of calibration.corner_points) m[p.name] = { x: p.x, y: p.y }
      setPicked(m)
    }
  }, [calibration])

  const pick = (name) => {
    setClickMode({
      type: 'calib', name,
      onPick: (x, y) => {
        setPicked((prev) => ({ ...prev, [name]: { x, y } }))
        setClickMode(null)
      },
    })
  }

  const suggest = async () => {
    setBusy(true)
    setError('')
    try {
      const r = await api.suggestCorners(videoId, 0)
      if (!r.found) setError('Auto-detect could not find a confident quadrilateral — click the corners manually.')
      else setPicked((prev) => ({ ...prev, ...Object.fromEntries(r.points.map((p) => [p.name, { x: p.x, y: p.y }])) }))
    } catch (e) { setError(e.message) }
    setBusy(false)
  }

  const save = async () => {
    const pts = Object.entries(picked).map(([name, p]) => ({ name, x: p.x, y: p.y }))
    if (pts.length < 4) { setError('Pick at least 4 reference points'); return }
    setBusy(true)
    setError('')
    try {
      await api.saveCalibration({ corner_points: pts, scope, video_id: videoId })
      onSaved?.()
    } catch (e) { setError(e.message) }
    setBusy(false)
  }

  const pickedCount = Object.keys(picked).length
  return (
    <div className="panel">
      <h2>Optional: court calibration {calibration?.calibrated
        ? <span className="pill human">calibrated ({calibration.scope})</span>
        : <span className="pill missed">not calibrated — Pass 2 needs this</span>}
      </h2>
      <div className="muted" style={{ marginBottom: 8 }}>
        One-time per camera setup — saved globally and reused by all future clips.
        Pause the video on a clear frame, select a point, then click its location in the video.
      </div>
      <div className="row" style={{ marginBottom: 8 }}>
        <button onClick={suggest} disabled={busy}>Auto-detect corners</button>
        <select value={scope} onChange={(e) => setScope(e.target.value)}>
          <option value="global">Save globally (default — camera is fixed)</option>
          <option value="clip_local">Clip-local (camera was bumped this day)</option>
        </select>
        <button className="blue" onClick={save} disabled={busy || pickedCount < 4}>
          Save calibration ({pickedCount} points)
        </button>
      </div>
      <div className="row" style={{ gap: 6 }}>
        {refPoints.map((rp) => {
          const active = clickMode?.type === 'calib' && clickMode.name === rp.name
          const has = picked[rp.name]
          return (
            <button key={rp.name}
                    className={active ? 'blue' : has ? 'primary' : ''}
                    onClick={() => pick(rp.name)}
                    title={has ? `${Math.round(has.x)}, ${Math.round(has.y)}` : 'not set'}>
              {active ? `click in video…` : `${has ? '✓ ' : ''}${rp.name.replaceAll('_', ' ')}`}
            </button>
          )
        })}
      </div>
      {error && <div className="error" style={{ marginTop: 6 }}>{error}</div>}
    </div>
  )
}
