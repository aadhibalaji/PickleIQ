import React, { useCallback, useEffect, useRef, useState } from 'react'
import Header from './components/Header.jsx'
import Footer from './components/Footer.jsx'
import ProcessingError from './components/ProcessingError.jsx'
import ProcessingProgress from './components/ProcessingProgress.jsx'
import RallyTimeline from './components/RallyTimeline.jsx'
import VideoPreview from './components/VideoPreview.jsx'
import { getPass1ProcessingState, retryPass1Processing } from './processingAdapter.js'
import useProcessingPolling from './useProcessingPolling.js'

const INITIAL_STATE = {
  status: 'queued', progress: null, message: '', previewUrl: null,
  previewKind: 'original-local', retainedRallies: [], fps: 30, error: null,
}

// Hands off to RallyReviewScreen once done — Pass 2 is never auto-queued
// here (see submitRallySelections in processingAdapter.js): the user decides
// what to keep first.
export default function Pass1Screen({ videoId, localFile, onComplete, onHome, onPastClips, onTesting }) {
  const headingRef = useRef(null)
  const playerRef = useRef(null)
  const [retrying, setRetrying] = useState(false)
  const loadState = useCallback((signal) => getPass1ProcessingState(videoId, signal), [videoId])
  const { processing, loadError: pollError, refresh } = useProcessingPolling(loadState, INITIAL_STATE)

  useEffect(() => { headingRef.current?.focus() }, [])

  useEffect(() => {
    if (processing.status === 'done' && processing.retainedRallies.length > 0) onComplete(processing)
  }, [processing, onComplete])

  const retry = async () => {
    if (retrying) return
    if (pollError && !processing.error) {
      refresh()
      return
    }
    try {
      setRetrying(true)
      await retryPass1Processing(videoId)
      refresh({ reset: true })
    } catch (error) {
      console.error('Pass 1 retry failed', error)
      refresh()
    } finally {
      setRetrying(false)
    }
  }

  const seek = (seconds) => {
    if (playerRef.current) playerRef.current.currentTime = seconds
  }
  const error = processing.error || pollError
  const zeroRallies = processing.status === 'done' && processing.retainedRallies.length === 0

  return (
    <div className="pickleiq-page">
      <Header onHome={onHome} onPastClips={onPastClips} onTesting={onTesting} />
      <main className="pickleiq-processing-main" aria-labelledby="pickleiq-pass1-title">
        <div className="pickleiq-processing-header">
          <button type="button" className="pickleiq-processing-back" onClick={onHome}>&larr; Home</button>
          <h1 ref={headingRef} tabIndex="-1" id="pickleiq-pass1-title">Clipping Video...</h1>
        </div>
        <div className="pickleiq-processing-divider" />
        {error ? (
          <ProcessingError message={error} onRetry={retry} onHome={onHome} retrying={retrying} />
        ) : zeroRallies ? (
          <ProcessingError title="No rallies were found"
            message="Pass 1 completed, but no rally sections were retained for further analysis."
            onRetry={retry} onHome={onHome} retrying={retrying} />
        ) : (
          <section aria-label="Pass 1 clipping progress">
            <ProcessingProgress label="Pass 1 of 3 · detecting rallies" progress={processing.progress}
              message={processing.message} />
            <VideoPreview previewUrl={processing.previewUrl} localFile={localFile}
              previewKind={processing.previewKind} playerRef={playerRef} />
            <RallyTimeline
              sections={processing.retainedRallies.map((rally) => ({
                id: rally.id, startFrame: rally.start_frame,
                endFrame: rally.end_frame, decision: 'kept',
              }))}
              fps={processing.fps} durationSeconds={processing.durationSeconds}
              onSeek={seek} emptyLabel="Scanning for rallies…" />
          </section>
        )}
      </main>
      <Footer left="PROCESSING" right="PASS 1 OF 3" />
    </div>
  )
}
