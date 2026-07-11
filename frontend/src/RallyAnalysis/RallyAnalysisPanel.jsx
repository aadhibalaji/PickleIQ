import React from 'react'
import ClipList from './ClipList.jsx'

export default function RallyAnalysisPanel({ videoId, onSeek }) {
  return (
    <div className="rally-analysis-panel">
      <h3>Rally Analysis</h3>
      <ClipList title="Highlights" items={[]} />
      <ClipList title="Coaching Moments" items={[]} />
    </div>
  )
}
