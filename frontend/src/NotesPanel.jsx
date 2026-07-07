import React, { useEffect, useRef, useState } from 'react'
import { api } from './api.js'

/*
Free-form problem notes (typed or voice) that feed the discrepancy analysis.
Scope routing: method-level feedback defaults to GLOBAL (it teaches the
system); mark a note clip-local when it's an oddity of this clip only.
Voice uses the browser's Web Speech API (Chrome/Edge) — transcribed
client-side, stored as text with source='voice'.
*/
export default function NotesPanel({ videoId, videoTime }) {
  const [notes, setNotes] = useState([])
  const [text, setText] = useState('')
  const [scope, setScope] = useState('global')
  const [listening, setListening] = useState(false)
  const recRef = useRef(null)

  const refresh = () => api.listNotes(videoId).then(setNotes).catch(() => {})
  useEffect(() => { refresh() }, [videoId]) // eslint-disable-line

  const add = async (body, source) => {
    if (!body.trim()) return
    await api.addNote(videoId, { text: body.trim(), source, scope, video_time: videoTime })
    setText('')
    refresh()
  }

  const speechSupported = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window
  const toggleVoice = () => {
    if (listening) { recRef.current?.stop(); return }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    const rec = new SR()
    rec.lang = 'en-US'
    rec.interimResults = false
    rec.onresult = (e) => {
      const t = Array.from(e.results).map((r) => r[0].transcript).join(' ')
      add(t, 'voice')
    }
    rec.onend = () => setListening(false)
    rec.onerror = () => setListening(false)
    recRef.current = rec
    setListening(true)
    rec.start()
  }

  return (
    <div className="panel">
      <h2>Notes about problems you see</h2>
      <div className="muted" style={{ marginBottom: 6 }}>
        Pause on the exact frame first — the current video position is attached to each note,
        which is what lets a specific detection failure be pinpointed and fixed. Examples:
        "ball not tracked on the far side here", "shoe of the far player is being marked as the
        ball", "this ball is clearly IN but got called out", "far player box jumps to the wrong
        person", "rally starts mid-downtime instead of just before the serve", "Pass 1 court
        size is off". These feed the report and the fixes.
      </div>
      <textarea value={text} onChange={(e) => setText(e.target.value)}
                placeholder={`Type a note… (attached to ${(videoTime ?? 0).toFixed(1)}s)`} />
      <div className="row" style={{ marginTop: 6 }}>
        <select value={scope} onChange={(e) => setScope(e.target.value)}>
          <option value="global">Global — about the method (default)</option>
          <option value="clip_local">Clip-local — oddity of this clip only</option>
        </select>
        <button className="blue" onClick={() => add(text, 'typed')} disabled={!text.trim()}>Add note</button>
        {speechSupported && (
          <button className={listening ? 'danger' : ''} onClick={toggleVoice}>
            {listening ? '● Recording… (click to stop)' : '🎤 Voice note'}
          </button>
        )}
      </div>
      {notes.length > 0 && (
        <table style={{ marginTop: 10 }}>
          <tbody>
            {notes.map((n) => (
              <tr key={n.id}>
                <td style={{ width: '60%' }}>{n.text}</td>
                <td><span className="pill">{n.source}</span> <span className={`pill ${n.scope === 'global' ? 'human' : ''}`}>{n.scope}</span></td>
                <td className="muted">{n.video_time != null ? `@${n.video_time.toFixed(1)}s` : ''}</td>
                <td><button className="danger" onClick={() => api.deleteNote(n.id).then(refresh)}>✕</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
