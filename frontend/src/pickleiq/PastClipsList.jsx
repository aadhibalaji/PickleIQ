import React, { useEffect, useState } from 'react'
import Header from './components/Header.jsx'
import Footer from './components/Footer.jsx'
import { listPastClips } from './processingAdapter.js'

function formatDate(uploadedAt) {
  if (!uploadedAt) return ''
  return new Date(uploadedAt * 1000).toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
  })
}

export default function PastClipsList({ onSelect, onHome, selectError, onTesting }) {
  const [clips, setClips] = useState(null) // null = loading
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    listPastClips()
      .then((list) => { if (!cancelled) setClips(list) })
      .catch((loadError) => { if (!cancelled) setError(loadError.message || 'Could not load past clips.') })
    return () => { cancelled = true }
  }, [])

  return (
    <div className="pickleiq-page">
      <Header active="pastclips" onHome={onHome} onPastClips={() => {}} onTesting={onTesting} />
      <main className="pickleiq-pastclips-main" aria-labelledby="pickleiq-history-title">
        <h1 id="pickleiq-history-title">Past Clips</h1>
        <div className="pickleiq-pastclips-divider" />
        <section aria-label="Past clips">
          {error && <p className="pickleiq-error">{error}</p>}
          {selectError && <p className="pickleiq-error">{selectError}</p>}
          {clips === null && !error && <p className="pickleiq-review-summary">Loading…</p>}
          {clips?.length === 0 && <p className="pickleiq-results-empty">No clips uploaded yet.</p>}
          {clips?.length > 0 && (
            <ul className="pickleiq-clip-list">
              {clips.map((clip) => (
                <li key={clip.id}>
                  <button type="button" className="pickleiq-clip-row" disabled={!clip.ready}
                    onClick={() => onSelect(clip.id, clip.status, clip.resumeStage)}>
                    <span className="pickleiq-clip-name">{clip.filename}</span>
                    <span className="pickleiq-clip-meta">
                      {formatDate(clip.uploadedAt)}
                      <span className={`pickleiq-clip-status${clip.ready ? ` is-${clip.status}` : ''}`}>
                        {clip.statusLabel}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          <button type="button" className="pickleiq-secondary-button pickleiq-history-back" onClick={onHome}>
            Back to Home
          </button>
        </section>
      </main>
      <Footer left={`${clips?.length ?? 0} CLIPS`} right="SORTED BY DATE" />
    </div>
  )
}
