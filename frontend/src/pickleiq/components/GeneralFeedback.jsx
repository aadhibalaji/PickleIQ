import React from 'react'

// "Player 2" -> "P2" fallback avatar when no photo was captured for a player.
function initials(playerLabel) {
  const match = playerLabel.match(/\d+/)
  return match ? `P${match[0]}` : playerLabel.slice(0, 2).toUpperCase()
}

export default function GeneralFeedback({ feedback }) {
  // No standalone "General feedback" heading in the redesign — the bordered
  // left panel it renders inside (see ResultsScreen) already reads as the
  // feedback panel. The label survives as a visually-hidden section name so
  // screen reader users still get an accessible landmark.
  if (!feedback || (Array.isArray(feedback) && feedback.length === 0)) {
    return (
      <section className="pickleiq-general-feedback" aria-labelledby="pickleiq-general-title">
        <p className="pickleiq-visually-hidden" id="pickleiq-general-title">General feedback</p>
        <p className="pickleiq-results-empty">General coaching feedback was not returned.</p>
      </section>
    )
  }

  return (
    <section className="pickleiq-general-feedback" aria-labelledby="pickleiq-general-title">
      <p className="pickleiq-visually-hidden" id="pickleiq-general-title">General feedback</p>
      {feedback.map((entry, index) => (
        <div className="pickleiq-general-entry" key={entry.id || index}>
          <div className="pickleiq-player-heading">
            {entry.photo
              ? <img className="pickleiq-player-photo" src={entry.photo} alt="" />
              : entry.playerLabel && (
                <span className="pickleiq-player-photo" aria-hidden="true">{initials(entry.playerLabel)}</span>
              )}
            {entry.playerLabel && <h2>{entry.playerLabel}</h2>}
          </div>
          {entry.summary && <p>{entry.summary}</p>}
          {entry.takeaways?.length > 0 && (
            <>
              <p className="pickleiq-takeaways-label">Key Takeaways</p>
              <ul className="pickleiq-takeaways-list">
                {entry.takeaways.map((point, i) => <li key={i}>{point}</li>)}
              </ul>
            </>
          )}
        </div>
      ))}
    </section>
  )
}
