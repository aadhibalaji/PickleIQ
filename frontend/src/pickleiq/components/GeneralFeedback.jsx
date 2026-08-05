import React from 'react'

function FeedbackText({ value }) {
  if (Array.isArray(value)) {
    return value.filter(Boolean).map((paragraph, index) => <p key={index}>{String(paragraph)}</p>)
  }
  return value ? <p>{String(value)}</p> : null
}

export default function GeneralFeedback({ feedback }) {
  if (!feedback || (Array.isArray(feedback) && feedback.length === 0)) {
    return (
      <section className="pickleiq-general-feedback" aria-labelledby="pickleiq-general-title">
        <p className="pickleiq-results-label" id="pickleiq-general-title">General feedback</p>
        <p className="pickleiq-results-empty">General coaching feedback was not returned.</p>
      </section>
    )
  }

  if (typeof feedback === 'string') {
    return (
      <section className="pickleiq-general-feedback" aria-labelledby="pickleiq-general-title">
        <p className="pickleiq-results-label" id="pickleiq-general-title">General feedback</p>
        <FeedbackText value={feedback} />
      </section>
    )
  }

  return (
    <section className="pickleiq-general-feedback" aria-labelledby="pickleiq-general-title">
      <p className="pickleiq-results-label" id="pickleiq-general-title">General feedback</p>
      {feedback.map((entry, index) => (
        <div className="pickleiq-general-entry" key={entry.id || index}>
          {entry.playerLabel && <h2>{entry.playerLabel}</h2>}
          <FeedbackText value={entry.praise} />
          <FeedbackText value={entry.improve} />
          <FeedbackText value={entry.theme} />
        </div>
      ))}
    </section>
  )
}
