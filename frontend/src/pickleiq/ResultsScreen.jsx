import React, { useMemo, useRef, useState } from 'react'
import Header from './components/Header.jsx'
import Footer from './components/Footer.jsx'
import GeneralFeedback from './components/GeneralFeedback.jsx'
import PrimaryButton from './components/PrimaryButton.jsx'
import TimestampFeedback from './components/TimestampFeedback.jsx'
import VideoPreview from './components/VideoPreview.jsx'
import { buildRallySegments, resolveFeedbackSeekSeconds } from './processingAdapter.js'

export default function ResultsScreen({ videoId, analysisResult, pass2Result, localFile, onHome, onPastClips, onTesting }) {
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
  // Every video this screen can show is the untrimmed original: either the
  // local File, or api.streamUrl, which serves that same upload (media.py
  // stream_video) — 'stream'/'pass1-output' are just labels for where the URL
  // came from, not different cuts. So the displayed timeline is 'source'
  // unless a real re-encode declares otherwise. The old gate ("no server URL
  // at all") silently disabled rally-only playback AND tip seeking for past
  // clips and resumed sessions, which stream instead of holding a File.
  const displayedTimeline = analysisResult?.finalVideoTimeline || 'source'
  const onSourceTimeline = displayedTimeline === 'source'

  const selectFeedback = (feedbackId) => {
    setSelectedFeedbackId(feedbackId)
    const entry = timestampFeedback.find((item) => item.id === feedbackId)
    if (!entry || !playerRef.current) return
    const seekSeconds = resolveFeedbackSeekSeconds(entry, displayedTimeline)
    if (seekSeconds !== null) playerRef.current.currentTime = seekSeconds
  }

  const playerCount = analysisResult?.generalFeedback?.length || 0

  return (
    <div className="pickleiq-page">
      <Header onHome={onHome} onPastClips={onPastClips} onTesting={onTesting} />
      <main className="pickleiq-results-main" aria-labelledby="pickleiq-results-title">
        <h1 id="pickleiq-results-title">Results</h1>
        <div className="pickleiq-results-divider" />
        <div className="pickleiq-results-layout">
          <div className="pickleiq-results-left">
            <GeneralFeedback feedback={analysisResult?.generalFeedback} />
          </div>
          <div className="pickleiq-results-right">
            <VideoPreview previewUrl={videoUrl} localFile={localFile}
              previewKind={videoKind} playerRef={playerRef}
              skipSegments={onSourceTimeline ? skipSegments : null}
              overlayVideoId={onSourceTimeline ? videoId : null}
              overlayFps={pass2Result?.fps || 30} />
            <TimestampFeedback entries={timestampFeedback} selectedId={selectedFeedbackId}
              onSelect={selectFeedback} />
          </div>
        </div>
        <div className="pickleiq-results-cta">
          <PrimaryButton onClick={onHome}>Back to Home</PrimaryButton>
        </div>
      </main>
      <Footer left="ANALYSIS COMPLETE" right={`${playerCount} PLAYERS`} />
    </div>
  )
}
