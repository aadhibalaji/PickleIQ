import React, { useMemo, useRef, useState } from 'react'
import GeneralFeedback from './components/GeneralFeedback.jsx'
import PrimaryButton from './components/PrimaryButton.jsx'
import TimestampFeedback from './components/TimestampFeedback.jsx'
import VideoPreview from './components/VideoPreview.jsx'
import { buildRallySegments, resolveFeedbackSeekSeconds } from './processingAdapter.js'

export default function ResultsScreen({ analysisResult, pass2Result, localFile, onHome }) {
  const [selectedFeedbackId, setSelectedFeedbackId] = useState(null)
  const playerRef = useRef(null)
  const timestampFeedback = useMemo(
    () => [...(Array.isArray(analysisResult?.timestampFeedback)
      ? analysisResult.timestampFeedback : [])]
      .sort((a, b) => a.sourceTimestampSeconds - b.sourceTimestampSeconds),
    [analysisResult?.timestampFeedback],
  )
  const videoUrl = analysisResult?.finalVideoUrl || pass2Result?.finalVideoUrl
    || pass2Result?.previewUrl || null
  const videoKind = analysisResult?.videoSourceKind || pass2Result?.previewKind || 'original-local'
  // "Cropped" playback without a re-encode: skip the dead time between kept
  // rallies on the original video. Only applies while the displayed video is
  // on the source timeline — a real trimmed export would need remapping.
  const keptSections = analysisResult?.keptSections || pass2Result?.keptSections || []
  const skipSegments = useMemo(
    () => buildRallySegments(keptSections, pass2Result?.fps),
    [keptSections, pass2Result?.fps],
  )
  const displaysOriginalTimeline = !videoUrl && videoKind === 'original-local'
  const displayedTimeline = displaysOriginalTimeline
    ? 'source'
    : analysisResult?.finalVideoTimeline || null

  const selectFeedback = (feedbackId) => {
    setSelectedFeedbackId(feedbackId)
    const entry = timestampFeedback.find((item) => item.id === feedbackId)
    if (!entry || !playerRef.current) return
    const seekSeconds = resolveFeedbackSeekSeconds(entry, displayedTimeline)
    if (seekSeconds !== null) playerRef.current.currentTime = seekSeconds
  }

  return (
    <main className="pickleiq-page pickleiq-results-page" aria-labelledby="pickleiq-results-title">
      <h1 id="pickleiq-results-title">Results</h1>
      <div className="pickleiq-results-layout">
        <div className="pickleiq-results-left">
          <GeneralFeedback feedback={analysisResult?.generalFeedback} />
        </div>
        <div className="pickleiq-results-right">
          <VideoPreview previewUrl={videoUrl} localFile={localFile}
            previewKind={videoKind} playerRef={playerRef}
            skipSegments={displaysOriginalTimeline ? skipSegments : null} />
          <TimestampFeedback entries={timestampFeedback} selectedId={selectedFeedbackId}
            onSelect={selectFeedback} />
        </div>
      </div>
      <PrimaryButton className="pickleiq-results-home" onClick={onHome}>Back to Home</PrimaryButton>
    </main>
  )
}
