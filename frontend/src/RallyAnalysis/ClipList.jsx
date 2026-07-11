import React from 'react'

// items: [{label, timestampS, onSeek}]
export default function ClipList({ title, items = [] }) {
  return (
    <div className="clip-list">
      <h4>{title}</h4>
      {items.length === 0 && <div className="clip-list-empty">No clips yet.</div>}
      <ul>
        {items.map((item, i) => (
          <li key={i}>
            <button className="clip-list-item" onClick={() => item.onSeek?.(item.timestampS)}>
              <span className="clip-list-label">{item.label}</span>
              <span className="clip-list-time">{formatTime(item.timestampS)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

function formatTime(s) {
  if (s == null) return ''
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}
