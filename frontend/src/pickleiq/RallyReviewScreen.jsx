import React, { useEffect, useRef, useState } from 'react'
import PrimaryButton from './components/PrimaryButton.jsx'
import ProcessingError from './components/ProcessingError.jsx'
import ProcessingProgress from './components/ProcessingProgress.jsx'
import RallyTimeline from './components/RallyTimeline.jsx'
import VideoPreview from './components/VideoPreview.jsx'
import { submitRallySelections } from './processingAdapter.js'

function formatTime(frame, fps) {
  const seconds = Math.max(0, Number(frame) / fps)
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds - m * 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

// Consumer-facing stand-in for the dev tool's Coaching Selections table:
// one rally at a time, one button press per decision, then an optional step
// to mark any rally Pass 1 missed. Ends by submitting the selections (see
// submitRallySelections) and queuing Pass 2.
export default function RallyReviewScreen({ videoId, localFile, pass1Result, onComplete, onHome }) {
  const candidates = pass1Result?.retainedRallies || []
  const fps = pass1Result?.fps || 30
  const playerRef = useRef(null)

  const [stage, setStage] = useState(candidates.length ? 'review' : 'draw') // 'review' | 'draw' | 'submitting'
  const [reviewIndex, setReviewIndex] = useState(0)
  const [keptIds, setKeptIds] = useState([])
  const [skippedIds, setSkippedIds] = useState([])
  const [drawnRanges, setDrawnRanges] = useState([])
  const [pendingStart, setPendingStart] = useState(null)
  const [drawError, setDrawError] = useState('')
  const [submitError, setSubmitError] = useState('')

  const decide = (keep) => {
    const candidate = candidates[reviewIndex]
    if (keep) setKeptIds((ids) => [...ids, candidate.id])
    else setSkippedIds((ids) => [...ids, candidate.id])
    const next = reviewIndex + 1
    if (next >= candidates.length) setStage('draw')
    else setReviewIndex(next)
  }

  const goBack = () => {
    if (reviewIndex === 0) return
    const prevIndex = reviewIndex - 1
    const prevId = candidates[prevIndex].id
    setKeptIds((ids) => ids.filter((id) => id !== prevId))
    setSkippedIds((ids) => ids.filter((id) => id !== prevId))
    setReviewIndex(prevIndex)
  }

  // Auto-seek the video to each candidate's start as you advance/go back —
  // if the <video> hasn't loaded its metadata yet (currentTime writes are
  // ignored until then), retry once metadata arrives.
  useEffect(() => {
    if (stage !== 'review') return
    const candidate = candidates[reviewIndex]
    const video = playerRef.current
    if (!candidate || !video) return
    const seek = () => { video.currentTime = candidate.start_frame / fps }
    if (video.readyState >= 1) {
      seek()
      return undefined
    }
    video.addEventListener('loadedmetadata', seek, { once: true })
    return () => video.removeEventListener('loadedmetadata', seek)
  }, [stage, reviewIndex, candidates, fps])

  const seekTo = (seconds) => { if (playerRef.current) playerRef.current.currentTime = seconds }

  const currentFrame = () => Math.round((playerRef.current?.currentTime || 0) * fps)
  const markStart = () => { setDrawError(''); setPendingStart(currentFrame()) }
  const markEnd = () => {
    const frame = currentFrame()
    if (pendingStart === null || frame <= pendingStart) {
      setDrawError('End must be after start.')
      return
    }
    setDrawError('')
    setDrawnRanges((ranges) => [...ranges, { start_frame: pendingStart, end_frame: frame }])
    setPendingStart(null)
  }
  const removeDrawn = (index) => setDrawnRanges((ranges) => ranges.filter((_, i) => i !== index))

  const totalSelected = keptIds.length + drawnRanges.length

  const submit = async () => {
    if (totalSelected === 0) {
      setDrawError('Keep at least one rally before continuing.')
      return
    }
    setStage('submitting')
    setSubmitError('')
    try {
      await submitRallySelections(videoId, { keptRallyIds: keptIds, drawnRanges })
      onComplete({ keptRallyIds: keptIds, drawnRanges })
    } catch (error) {
      setSubmitError(error.message || 'Could not start Pass 2.')
    }
  }

  if (stage === 'submitting') {
    return (
      <main className="pickleiq-page pickleiq-processing-page" aria-labelledby="pickleiq-review-title">
        <h1 id="pickleiq-review-title">Starting Pass 2...</h1>
        {submitError ? (
          <ProcessingError message={submitError} onRetry={submit} onHome={onHome} retryLabel="Retry" />
        ) : (
          <section className="pickleiq-processing-content" aria-label="Starting Pass 2">
            <ProcessingProgress label="Pass 2" progress={null} message="Queuing your selected rallies…" />
          </section>
        )}
      </main>
    )
  }

  if (stage === 'review') {
    const candidate = candidates[reviewIndex]
    const timelineSections = candidates.map((c, i) => ({
      id: c.id, startFrame: c.start_frame, endFrame: c.end_frame,
      decision: i === reviewIndex ? 'current'
        : keptIds.includes(c.id) ? 'kept'
        : skippedIds.includes(c.id) ? 'rejected' : 'unknown',
    }))
    return (
      <main className="pickleiq-page pickleiq-processing-page" aria-labelledby="pickleiq-review-title">
        <h1 id="pickleiq-review-title">Review Rallies</h1>
        <section className="pickleiq-processing-content" aria-label={`Rally ${reviewIndex + 1} of ${candidates.length}`}>
          <p className="pickleiq-review-count">Rally {reviewIndex + 1} of {candidates.length}</p>
          <div className="pickleiq-review-card">
            <p className="pickleiq-review-time">
              {formatTime(candidate.start_frame, fps)} – {formatTime(candidate.end_frame, fps)}
            </p>
            <p className="pickleiq-review-duration">
              {formatTime(candidate.end_frame - candidate.start_frame, fps)} long
            </p>
          </div>
          <VideoPreview localFile={localFile} previewKind="original-local" playerRef={playerRef} />
          <RallyTimeline sections={timelineSections} fps={fps} durationSeconds={pass1Result?.durationSeconds}
            onSeek={seekTo} emptyLabel="No rallies detected" />
          <div className="pickleiq-review-actions">
            <button type="button" className="pickleiq-secondary-button" onClick={() => decide(false)}>
              Skip
            </button>
            <PrimaryButton onClick={() => decide(true)}>Keep</PrimaryButton>
          </div>
          {reviewIndex > 0 && (
            <button type="button" className="pickleiq-review-back" onClick={goBack}>← back</button>
          )}
        </section>
      </main>
    )
  }

  // stage === 'draw'
  return (
    <main className="pickleiq-page pickleiq-processing-page" aria-labelledby="pickleiq-review-title">
      <h1 id="pickleiq-review-title">Add a Missed Rally?</h1>
      <section className="pickleiq-processing-content" aria-label="Mark a missed rally">
        <p className="pickleiq-review-summary">{keptIds.length} kept so far.</p>
        <VideoPreview localFile={localFile} previewKind="original-local" playerRef={playerRef} />
        <div className="pickleiq-review-draw-actions">
          <PrimaryButton onClick={markStart} disabled={pendingStart !== null}>
            Mark start
          </PrimaryButton>
          <PrimaryButton onClick={markEnd} disabled={pendingStart === null}>
            Mark end
          </PrimaryButton>
        </div>
        {pendingStart !== null && (
          <p className="pickleiq-review-summary">Range open since {formatTime(pendingStart, fps)} — scrub to the end and tap "Mark end".</p>
        )}
        {drawnRanges.map((range, i) => (
          <div key={i} className="pickleiq-review-drawn-row">
            <span>{formatTime(range.start_frame, fps)} – {formatTime(range.end_frame, fps)}</span>
            <button type="button" className="pickleiq-secondary-button" onClick={() => removeDrawn(i)}>Remove</button>
          </div>
        ))}
        {drawError && <p className="pickleiq-error" style={{ position: 'static' }}>{drawError}</p>}
        <PrimaryButton className="pickleiq-start-button-inline" onClick={submit} disabled={totalSelected === 0}>
          Start Pass 2
        </PrimaryButton>
      </section>
    </main>
  )
}
