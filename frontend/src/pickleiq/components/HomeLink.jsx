import React from 'react'

// Lets someone bail out of a clip mid-pipeline (upload already started,
// review not finished, still processing, whatever) and get back to the
// home screen — the backend job keeps running regardless, and thanks to
// Past Clips' "needs review"/"still processing" states, nothing is lost
// except any review decisions not yet submitted via "Start Pass 2".
export default function HomeLink({ onHome }) {
  return (
    <button type="button" className="pickleiq-secondary-button pickleiq-home-link" onClick={onHome}>
      ← Home
    </button>
  )
}
