import React, { useEffect, useState } from 'react'
import Header from './components/Header.jsx'
import Footer from './components/Footer.jsx'
import { api } from '../api.js'
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
  const [deletingId, setDeletingId] = useState(null)
  const [deleteError, setDeleteError] = useState('')

  useEffect(() => {
    let cancelled = false
    listPastClips()
      .then((list) => { if (!cancelled) setClips(list) })
      .catch((loadError) => { if (!cancelled) setError(loadError.message || 'Could not load past clips.') })
    return () => { cancelled = true }
  }, [])

  // Cancels the job wherever it's actually running (this container or, via
  // an S3 relay, the EC2 worker — see api/media.py's DELETE /videos/{id})
  // and wipes every trace of the clip: DB rows and S3 data both. Real,
  // irreversible deletion, hence the confirm.
  const deleteClip = async (clip) => {
    if (!window.confirm(`Delete "${clip.filename}"? This can't be undone.`)) return
    setDeleteError('')
    setDeletingId(clip.id)
    try {
      await api.deleteVideo(clip.id)
      setClips((list) => list?.filter((c) => c.id !== clip.id) ?? list)
    } catch (deleteErr) {
      setDeleteError(deleteErr.message || 'Could not delete that clip.')
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="pickleiq-page">
      <Header active="pastclips" onHome={onHome} onPastClips={() => {}} onTesting={onTesting} />
      <main className="pickleiq-pastclips-main" aria-labelledby="pickleiq-history-title">
        <h1 id="pickleiq-history-title">Past Clips</h1>
        <div className="pickleiq-pastclips-divider" />
        <section aria-label="Past clips">
          {error && <p className="pickleiq-error">{error}</p>}
          {selectError && <p className="pickleiq-error">{selectError}</p>}
          {deleteError && <p className="pickleiq-error">{deleteError}</p>}
          {clips === null && !error && <p className="pickleiq-review-summary">Loading…</p>}
          {clips?.length === 0 && <p className="pickleiq-results-empty">No clips uploaded yet.</p>}
          {clips?.length > 0 && (
            <ul className="pickleiq-clip-list">
              {clips.map((clip) => (
                <li key={clip.id} className="pickleiq-clip-row">
                  <button type="button" className="pickleiq-clip-row-main" disabled={!clip.ready}
                    onClick={() => onSelect(clip.id, clip.status, clip.resumeStage)}>
                    <span className="pickleiq-clip-name">{clip.filename}</span>
                    <span className="pickleiq-clip-meta">
                      {formatDate(clip.uploadedAt)}
                      <span className={`pickleiq-clip-status${clip.ready ? ` is-${clip.status}` : ''}`}>
                        {clip.statusLabel}
                      </span>
                    </span>
                  </button>
                  <button type="button" className="pickleiq-clip-delete"
                    disabled={deletingId === clip.id} aria-label={`Delete ${clip.filename}`}
                    onClick={() => deleteClip(clip)}>
                    {deletingId === clip.id ? 'Deleting…' : 'Delete'}
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
