import React from 'react'

export default function GeneralFeedback({ feedback }) {
  if (!feedback || (Array.isArray(feedback) && feedback.length === 0)) {
    return (
      <section className="pickleiq-general-feedback" aria-labelledby="pickleiq-general-title">
        <p className="pickleiq-results-label" id="pickleiq-general-title">General feedback</p>
        <p className="pickleiq-results-empty">General coaching feedback was not returned.</p>
      </section>
    )
  }

  return (
    <section className="pickleiq-general-feedback" aria-labelledby="pickleiq-general-title">
      <p className="pickleiq-results-label" id="pickleiq-general-title">General feedback</p>
      {feedback.map((entry, index) => (
        <div className="pickleiq-general-entry" key={entry.id || index}>
          <div className="pickleiq-player-heading">
            {entry.photo && <img className="pickleiq-player-photo" src={entry.photo} alt="" />}
            {entry.playerLabel && <h2>{entry.playerLabel}</h2>}
          </div>
          {entry.summary && <p>{entry.summary}</p>}
          {entry.takeaways?.length > 0 && (
            <>
              <p className="pickleiq-takeaways-label">Key Takeaways:</p>
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
