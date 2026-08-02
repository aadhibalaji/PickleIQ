import React, { useEffect, useState } from 'react'
import { api } from './api.js'
import Workspace from './Workspace.jsx'
import Home from './Home.jsx'

export default function App() {
  const [view, setView] = useState('app') // 'app' | 'testing'
  const [videos, setVideos] = useState([])
  const [current, setCurrent] = useState(null)
  const [uploading, setUploading] = useState(null) // progress 0..1
  const [error, setError] = useState('')

  const refresh = () => api.listVideos().then(setVideos).catch((e) => setError(e.message))
  useEffect(() => { refresh() }, [])

  const onUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError('')
    setUploading(0)
    try {
      // Pass 1 starts in the background immediately — jump straight into the
      // workspace so labeling can begin while it runs.
      const v = await api.uploadVideo(file, setUploading)
      setUploading(null)
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
      <b>🏓 Pickleball Rally Detection</b>
      <div className="row">
        <button
          onClick={() => setView('app')}
          style={{ fontWeight: view === 'app' ? 'bold' : 'normal', textDecoration: view === 'app' ? 'underline' : 'none' }}
        >
          App
        </button>
        <button
          onClick={() => setView('testing')}
          style={{ fontWeight: view === 'testing' ? 'bold' : 'normal', textDecoration: view === 'testing' ? 'underline' : 'none' }}
        >
          Testing
        </button>
      </div>
    </div>
  )

  if (view === 'app') {
    return (
      <div>
        {nav}
        <Home videos={videos} current={current} onOpenTesting={() => setView('testing')} />
      </div>
    )
  }

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
      <h1>🏓 Pickleball Rally Detection</h1>
      <div className="panel">
        <div className="row spread">
          <div>
            <label className="row">
              <input type="file" accept="video/*" onChange={onUpload} disabled={uploading !== null} />
            </label>
            <div className="muted">
              Upload a fixed-camera clip. Pass 1 processing starts immediately in the
              background — you can label rallies while it runs.
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
