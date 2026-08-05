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
          return (
            <button key={entry.id} type="button" aria-pressed={active}
              disabled={!available} className={active ? 'is-selected' : ''}
              onClick={() => onSelect(entry.id)}
              aria-label={`${entry.displayLabel}. ${available ? 'Show coaching feedback' : 'Feedback unavailable'}`}>
              {entry.displayLabel}
            </button>
          )
        })}
      </div>
      {selected?.feedback && (
        <section className="pickleiq-selected-feedback" aria-live="polite" aria-atomic="true">
          <p>{selected.displayLabel}</p>
          <div>{selected.feedback}</div>
        </section>
      )}
    </div>
  )
}
