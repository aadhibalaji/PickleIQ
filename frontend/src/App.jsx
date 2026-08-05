import React, { useEffect, useState } from 'react'
import { api } from './api.js'
import Workspace from './Workspace.jsx'
import CoachingSetupSelector from './RallyAnalysis/CoachingSetupSelector.jsx'

export default function App() {
  const [videos, setVideos] = useState([])
  const [current, setCurrent] = useState(null)
  const [uploading, setUploading] = useState(null) // progress 0..1
  const [error, setError] = useState('')
  const [matchType, setMatchType] = useState('') // required at upload — see api/media.py
  // Coaching preferences chosen BEFORE upload (feedback focus) — PATCHed
  // onto the video as soon as the upload returns an id. Null = user didn't
  // choose; the backend defaults focus to form.
  const [clipPrefs, setClipPrefs] = useState({ match_type: null, coaching_focus: null })

  const refresh = () => api.listVideos().then(setVideos).catch((e) => setError(e.message))
  useEffect(() => { refresh() }, [])

  const onUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!matchType) {
      setError('Select singles or doubles before uploading.')
      e.target.value = ''
      return
    }
    setError('')
    setUploading(0)
    try {
      // Pass 1 starts in the background immediately — jump straight into the
      // workspace so labeling can begin while it runs.
      const v = await api.uploadVideo(file, matchType, setUploading)
      const prefs = Object.fromEntries(
        Object.entries(clipPrefs).filter(([, val]) => val != null))
      if (Object.keys(prefs).length) {
        // Preferences are a nice-to-have on the upload path — never fail the
        // upload over them; they stay editable in the Rally Analysis tab.
        await api.updateVideo(v.id, prefs).catch(() => {})
      }
      setUploading(null)
      setMatchType('')
      setCurrent(v.id)
      refresh()
    } catch (err) {
      setUploading(null)
      setError(err.message)
    }
    e.target.value = ''
  }

  const onDelete = async (e, v) => {
    e.stopPropagation()
    if (!window.confirm(`Delete "${v.filename}" and all its data? This can't be undone.`)) return
    setError('')
    try {
      await api.deleteVideo(v.id)
      refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  const nav = (
    <div className="row spread" style={{ padding: '8px 16px', borderBottom: '1px solid #ccc' }}>
      <b>Pickleball Rally Detection</b>
    </div>
  )

  if (current) {
    return (
      <div>
        {nav}
        <Workspace videoId={current} onBack={() => { setCurrent(null); refresh() }} />
      </div>
    )
  }

  return (
    <div>
      {nav}
      <h1>Pickleball Rally Detection</h1>
      <div className="panel">
        <div className="row spread">
          <div>
            <label className="row">
              Match type:
              <select value={matchType} onChange={(e) => setMatchType(e.target.value)} disabled={uploading !== null}>
                <option value="">Select…</option>
                <option value="singles">Singles</option>
                <option value="doubles">Doubles</option>
              </select>
            </label>
            <label className="row">
              <input type="file" accept="video/*" onChange={onUpload}
                     disabled={uploading !== null || !matchType} />
            </label>
            <div style={{ margin: '8px 0' }}>
              <CoachingSetupSelector
                video={clipPrefs}
                onUpdateVideo={(fields) => setClipPrefs((prev) => ({ ...prev, ...fields }))} />
            </div>
            <div className="muted">
              Upload a fixed-camera clip. Pass 1 processing starts immediately in the
              background — you can label rallies while it runs. Match type is required —
              it caps how many players the tracker will identify per side.
            </div>
          </div>
          {uploading !== null && (
            <div className="row">
              <progress value={uploading} max="1" />
              <span className="muted">{Math.round(uploading * 100)}%</span>
            </div>
          )}
        </div>
        {error && <div className="error">{error}</div>}
      </div>

      <h2>Clips</h2>
      {videos.length === 0 && <div className="muted">No clips yet.</div>}
      {videos.map((v) => (
        <div key={v.id} className="panel row spread" style={{ cursor: 'pointer' }}
             onClick={() => setCurrent(v.id)}>
          <div>
            <b>{v.filename}</b>
            <div className="muted">
              {v.width}x{v.height} · {v.fps?.toFixed(1)} fps · {Math.round((v.frame_count || 0) / (v.fps || 30))}s
              · uploaded {new Date(v.uploaded_at * 1000).toLocaleString()}
            </div>
          </div>
          <div className="row">
            <button className="blue">Open →</button>
            <button className="danger" onClick={(e) => onDelete(e, v)}>Delete</button>
          </div>
        </div>
      ))}
    </div>
  )
}
