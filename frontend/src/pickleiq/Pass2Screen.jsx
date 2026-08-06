import React, { useCallback, useEffect, useRef, useState } from 'react'
import Header from './components/Header.jsx'
import Footer from './components/Footer.jsx'
import ProcessingError from './components/ProcessingError.jsx'
import ProcessingProgress from './components/ProcessingProgress.jsx'
import RallyTimeline from './components/RallyTimeline.jsx'
import VideoPreview from './components/VideoPreview.jsx'
import { getPass2ProcessingState, retryPass2Processing } from './processingAdapter.js'
import useProcessingPolling from './useProcessingPolling.js'

const INITIAL_STATE = {
  jobId: null, status: 'queued', progress: null, message: '', previewUrl: null,
  previewKind: 'original-local', sections: [], keptSections: [], rejectedSections: [], unknownSections: [],
  classificationComplete: false, finalVideoUrl: null, fps: 30, partial: false, error: null,
}

export default function Pass2Screen({ videoId, localFile, pass1Result, onComplete, onHome, onPastClips, onTesting }) {
  const headingRef = useRef(null)
  const playerRef = useRef(null)
  const [retrying, setRetrying] = useState(false)
  const loadState = useCallback(
    (signal) => getPass2ProcessingState(videoId, pass1Result, signal),
    [videoId, pass1Result],
  )
  const { processing, loadError, refresh } = useProcessingPolling(loadState, INITIAL_STATE)

  useEffect(() => { headingRef.current?.focus() }, [])

  useEffect(() => {
    if (processing.status === 'done' && processing.classificationComplete
      && processing.keptSections.length > 0) {
      onComplete(processing)
    }
  }, [processing, onComplete])

  const retry = async () => {
    if (retrying) return
    if (loadError && !processing.error) {
      refresh()
      return
    }
    try {
      setRetrying(true)
      await retryPass2Processing(videoId)
      refresh({ reset: true })
    } catch (error) {
      console.error('Pass 2 retry failed', error)
      refresh()
    } finally {
      setRetrying(false)
    }
  }

  const seek = (seconds) => {
    if (playerRef.current) playerRef.current.currentTime = seconds
  }
  const zeroKept = processing.status === 'done' && processing.classificationComplete
    && processing.keptSections.length === 0
  const error = processing.error || loadError

  return (
    <div className="pickleiq-page">
      <Header onHome={onHome} onPastClips={onPastClips} onTesting={onTesting} />
      <main className="pickleiq-processing-main" aria-labelledby="pickleiq-pass2-title">
        <div className="pickleiq-processing-header">
          <button type="button" className="pickleiq-processing-back" onClick={onHome}>&larr; Home</button>
          <h1 ref={headingRef} tabIndex="-1" id="pickleiq-pass2-title">Clipping Video...</h1>
        </div>
        <div className="pickleiq-processing-divider" />
        {error ? (
          <ProcessingError message={error} onRetry={retry} onHome={onHome}
            retryLabel="Retry Pass 2" retrying={retrying} />
        ) : zeroKept ? (
          <ProcessingError title="No usable rallies remained"
            message="Pass 2 reviewed every candidate, but none were suitable for player analysis."
            onRetry={retry} onHome={onHome} retryLabel="Retry Pass 2" retrying={retrying} />
        ) : (
          <section aria-label="Pass 2 clipping progress">
            <ProcessingProgress label="Pass 2 of 3 · classifying rallies" progress={processing.progress}
              message={processing.message} />
            <VideoPreview previewUrl={processing.previewUrl} localFile={localFile}
              previewKind={processing.previewKind} playerRef={playerRef} />
            <RallyTimeline sections={processing.sections} fps={processing.fps}
              durationSeconds={processing.durationSeconds} onSeek={seek}
              emptyLabel="Reviewing detected rallies…" />
          </section>
        )}
      </main>
      <Footer left="PROCESSING" right="PASS 2 OF 3" />
    </div>
  )
}
