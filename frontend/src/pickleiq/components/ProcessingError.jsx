import React from 'react'
import PrimaryButton from './PrimaryButton.jsx'

export default function ProcessingError({ message, onRetry, onHome, title = 'Clipping stopped',
  retryLabel = 'Retry Pass 1', inline = false, retrying = false }) {
  return (
    <section className={`pickleiq-processing-error${inline ? ' is-inline' : ''}`}
      role="alert" aria-labelledby="pickleiq-error-title">
      <h2 id="pickleiq-error-title">{title}</h2>
      <p>{message}</p>
      <div>
        {onRetry && <PrimaryButton onClick={onRetry} disabled={retrying} aria-busy={retrying}>
          {retrying ? 'Retrying…' : retryLabel}
        </PrimaryButton>}
        <button type="button" className="pickleiq-secondary-button" onClick={onHome}>Return home</button>
      </div>
    </section>
  )
}
