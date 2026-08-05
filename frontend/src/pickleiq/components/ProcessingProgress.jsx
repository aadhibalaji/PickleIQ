import React from 'react'

export default function ProcessingProgress({ label, progress, message, inactive = false }) {
  const determinate = Number.isFinite(progress)
  const percent = determinate ? Math.round(progress * 100) : null

  return (
    <div className="pickleiq-progress-block">
      <div className="pickleiq-progress-copy">
        <span>{label}</span>
        <span>{inactive ? 'Stopped' : determinate ? `${percent}%` : message || 'Processing'}</span>
      </div>
      {inactive ? (
        <div className="pickleiq-progress-track is-inactive" aria-label={`${label} stopped`}>
          <span style={{ width: determinate ? `${percent}%` : '0%' }} />
        </div>
      ) : determinate ? (
        <div className="pickleiq-progress-track" role="progressbar" aria-label={label}
          aria-valuemin="0" aria-valuemax="100" aria-valuenow={percent}>
          <span style={{ width: `${percent}%` }} />
        </div>
      ) : (
        <div className="pickleiq-progress-track is-indeterminate" role="progressbar"
          aria-label={`${label}: ${message || 'processing'}`}>
          <span />
        </div>
      )}
    </div>
  )
}
