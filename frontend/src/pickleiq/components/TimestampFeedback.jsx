import React from 'react'

export default function TimestampFeedback({ entries, selectedId, onSelect }) {
  if (!entries.length) {
    return <p className="pickleiq-results-empty pickleiq-timestamps-empty">No timestamp-specific feedback was returned.</p>
  }

  const selected = entries.find((entry) => entry.id === selectedId) || null
  return (
    <div className="pickleiq-timestamp-feedback">
      <div className="pickleiq-timestamp-list" aria-label="Timestamp-specific feedback">
        {entries.map((entry) => {
          const active = entry.id === selectedId
          const available = Boolean(entry.feedback)
          const label = entry.playerLabel ? `${entry.playerLabel} · ${entry.displayLabel}` : entry.displayLabel
          return (
            <button key={entry.id} type="button" aria-pressed={active}
              disabled={!available} className={active ? 'is-selected' : ''}
              onClick={() => onSelect(entry.id)}
              aria-label={`${label}. ${available ? 'Show coaching feedback' : 'Feedback unavailable'}`}>
              {entry.playerPhoto && <img className="pickleiq-player-photo pickleiq-player-photo--tiny" src={entry.playerPhoto} alt="" />}
              {entry.playerLabel && <span className="pickleiq-timestamp-player">{entry.playerLabel}</span>}
              {entry.displayLabel}
            </button>
          )
        })}
      </div>
      {selected?.feedback && (
        <section className="pickleiq-selected-feedback" aria-live="polite" aria-atomic="true">
          <div className="pickleiq-player-heading">
            {selected.playerPhoto && <img className="pickleiq-player-photo" src={selected.playerPhoto} alt="" />}
            <p>
              {selected.playerLabel && <span className="pickleiq-selected-player">{selected.playerLabel}</span>}
              {selected.displayLabel}
            </p>
          </div>
          <div>{selected.feedback}</div>
        </section>
      )}
    </div>
  )
}
