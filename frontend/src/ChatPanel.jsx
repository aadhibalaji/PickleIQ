import React, { useEffect, useRef, useState } from 'react'
import { api } from './api.js'

/*
In-app assistant. Same Claude model that works in the terminal, with full
context of this clip. It can read the feature traces, tune global parameters,
trigger reprocessing, and save handoff notes — everything the UI can do —
but it cannot edit CV source code (that stays with the coding agent). When
it takes an action, onStateChange refreshes the rest of the app.
*/
const SUGGESTIONS = [
  'Why did rally 4 get missed? Look at the feature trace.',
  'A ball that was clearly in got called out around 1:40 — check it.',
  'Compare Pass 2 to my labels and tell me the biggest problem.',
  'The far-side ball isn’t being tracked. What are my options?',
]

export default function ChatPanel({ videoId, fps, currentFrame, onStateChange }) {
  const [msgs, setMsgs] = useState([])   // {role, text}
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [actions, setActions] = useState([])
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight)
  }, [msgs, busy])

  const send = async (text) => {
    const t = (text ?? input).trim()
    if (!t || busy) return
    setError('')
    setInput('')
    const next = [...msgs, { role: 'user', text: t }]
    setMsgs(next)
    setBusy(true)
    try {
      const r = await api.chat(videoId, next)
      setMsgs([...next, { role: 'assistant', text: r.reply }])
      setActions(r.actions || [])
      if (r.state_changed) onStateChange?.()
    } catch (e) {
      setError(e.message)
      setMsgs(next) // keep the user's message; let them retry
    }
    setBusy(false)
  }

  const insertTimestamp = () => {
    const s = currentFrame / (fps || 30)
    const mm = Math.floor(s / 60)
    const stamp = `${mm}:${(s - mm * 60).toFixed(1).padStart(4, '0')} (frame ${currentFrame})`
    setInput((v) => `${v}${v && !v.endsWith(' ') ? ' ' : ''}at ${stamp} `)
  }

  return (
    <div>
      <div className="row spread">
        <div className="muted">
          The same Claude that works in your terminal, with full context of this clip. It can read
          the per-frame feature traces, tune global parameters, reprocess, and save handoff notes.
          It can’t edit the CV source code — for that it writes a precise note for the coding agent.
        </div>
        <button onClick={insertTimestamp} title="Insert the current video position">+ timestamp</button>
      </div>

      <div ref={scrollRef} style={{
        margin: '10px 0', maxHeight: 420, overflow: 'auto', display: 'flex',
        flexDirection: 'column', gap: 8,
      }}>
        {msgs.length === 0 && (
          <div className="row" style={{ flexWrap: 'wrap' }}>
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => send(s)} disabled={busy}
                      style={{ fontSize: 12, textAlign: 'left' }}>{s}</button>
            ))}
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '85%', background: m.role === 'user' ? 'var(--panel2)' : 'var(--panel)',
            border: '1px solid var(--border)', borderRadius: 10, padding: '8px 12px',
            whiteSpace: 'pre-wrap', fontSize: 13,
          }}>
            <div className="muted" style={{ fontSize: 11, marginBottom: 2 }}>
              {m.role === 'user' ? 'You' : 'Assistant'}
            </div>
            {m.text}
          </div>
        ))}
        {busy && (
          <div style={{ alignSelf: 'flex-start', maxWidth: '85%' }}>
            <div className="row"><div className="loadingbar" style={{ minWidth: 160 }} />
              <span className="muted">thinking / running tools…</span></div>
          </div>
        )}
      </div>

      {actions.length > 0 && (
        <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
          Last turn actions: {actions.map((a, i) => (
            <span key={i} className="pill" style={{ marginRight: 4 }}>🔧 {a.tool}</span>
          ))}
        </div>
      )}
      {error && <div className="error">{error}</div>}

      <div className="row">
        <textarea value={input} onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send() }}
                  placeholder="Ask about a detection, request a tuning change, or describe a problem… (Ctrl/Cmd+Enter to send)"
                  style={{ flex: 1 }} />
        <button className="primary" onClick={() => send()} disabled={busy || !input.trim()}>Send</button>
      </div>
    </div>
  )
}
