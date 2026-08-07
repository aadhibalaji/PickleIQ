import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Header from './components/Header.jsx'
import Footer from './components/Footer.jsx'
import ProcessingError from './components/ProcessingError.jsx'
import ProcessingProgress from './components/ProcessingProgress.jsx'
import TimeRemaining from './components/TimeRemaining.jsx'
import VideoPreview from './components/VideoPreview.jsx'
import {
  buildRallySegments,
  estimateAnalysisTotalSeconds,
  generatePlayerAnalysis,
  getAnalysisPreparationState,
  remainingAnalysisSeconds,
  retryAnalysisPreparation,
} from './processingAdapter.js'
import useProcessingPolling from './useProcessingPolling.js'

const PREPARATION_INITIAL = {
  jobId: null, status: 'queued', progress: null, message: '',
  estimatedSecondsRemaining: null, error: null,
}
const GENERATION_INITIAL = {
  status: 'idle', progress: null, estimatedSecondsRemaining: null, error: null,
}

export default function AnalysisScreen({ videoId, localFile, pass2Result, finalVideoUrl,
  onComplete, onHome, onPastClips, onTesting }) {
  const headingRef = useRef(null)
  const generationStartedRef = useRef(false)
  const generationControllerRef = useRef(null)
  const [generation, setGeneration] = useState(GENERATION_INITIAL)
  const [preparationRecovered, setPreparationRecovered] = useState(false)
  const loadState = useCallback((signal) => getAnalysisPreparationState(videoId, signal), [videoId])
  const { processing: preparation, loadError, refresh } = useProcessingPolling(
    loadState, PREPARATION_INITIAL,
  )
  const preservedVideoUrl = finalVideoUrl || pass2Result?.finalVideoUrl
    || pass2Result?.previewUrl || null
  const previewKind = pass2Result?.previewKind || 'original-local'
  const keptSections = pass2Result?.keptSections || []
  const skipSegments = useMemo(
    () => buildRallySegments(keptSections, pass2Result?.fps),
    [keptSections, pass2Result?.fps],
  )

  // Countdown estimate for the whole analysis stage (contacts prep + one
  // long coaching LLM call) — neither reports usable progress, so this is a
  // heuristic total scaled by rally count, ticking down once per second.
  const [analysisStartedAt, setAnalysisStartedAt] = useState(() => Date.now())
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  useEffect(() => {
    const timer = window.setInterval(
      () => setElapsedSeconds((Date.now() - analysisStartedAt) / 1000), 1000)
    return () => window.clearInterval(timer)
  }, [analysisStartedAt])
  const estimatedTotal = useMemo(
    () => estimateAnalysisTotalSeconds(keptSections.length), [keptSections.length])

  useEffect(() => { headingRef.current?.focus() }, [])
  useEffect(() => () => generationControllerRef.current?.abort(), [])

  const runGeneration = useCallback(async ({ retryContacts = false } = {}) => {
    if (generationStartedRef.current) return
    generationStartedRef.current = true
    const controller = new AbortController()
    generationControllerRef.current = controller
    setGeneration({
      status: 'processing', progress: null, estimatedSecondsRemaining: null, error: null,
    })
    try {
      if (retryContacts) {
        await retryAnalysisPreparation(videoId, controller.signal)
        setPreparationRecovered(true)
      }
      const result = await generatePlayerAnalysis(videoId, preservedVideoUrl, controller.signal)
      setGeneration(result)
    } catch (error) {
      if (error.name === 'AbortError') return
      setGeneration({
        status: 'failed', progress: null, estimatedSecondsRemaining: null,
        error: error.message || 'Player analysis could not be completed.',
      })
    } finally {
      generationStartedRef.current = false
      if (generationControllerRef.current === controller) generationControllerRef.current = null
    }
  }, [videoId, preservedVideoUrl])

  useEffect(() => {
    if (preparation.status === 'ready' && generation.status === 'idle') runGeneration()
  }, [preparation.status, generation.status, runGeneration])

  useEffect(() => {
    if (generation.status === 'completed') onComplete(generation)
  }, [generation, onComplete])

  const retry = () => {
    if (loadError && !preparation.error) {
      refresh()
      return
    }
    setGeneration(GENERATION_INITIAL)
    setAnalysisStartedAt(Date.now())
    setElapsedSeconds(0)
    runGeneration({ retryContacts: preparation.status === 'failed' && !preparationRecovered })
  }

  const generationActive = generation.status === 'processing'
  const error = generation.error
    || (!generationActive && !preparationRecovered ? preparation.error || loadError : null)
  const progress = generationActive ? generation.progress : preparation.progress
  const estimate = error ? null : remainingAnalysisSeconds(estimatedTotal, elapsedSeconds)
  const message = generationActive
    ? 'Generating coaching feedback'
    : preparation.message || (preparation.status === 'queued'
      ? 'Waiting to analyze players' : 'Analyzing player movement')

  return (
    <div className="pickleiq-page">
      <Header onHome={onHome} onPastClips={onPastClips} onTesting={onTesting} />
      <main className="pickleiq-processing-main" aria-labelledby="pickleiq-analysis-title">
        <div className="pickleiq-processing-header">
          <button type="button" className="pickleiq-processing-back" onClick={onHome}>&larr; Home</button>
          <h1 ref={headingRef} tabIndex="-1" id="pickleiq-analysis-title">Analyzing players</h1>
        </div>
        <div className="pickleiq-processing-divider" />
        <section aria-label="Player analysis progress">
          <TimeRemaining seconds={estimate} />
          <ProcessingProgress label="Pass 3 of 3 · analyzing players" progress={progress} message={message}
            inactive={Boolean(error)} />
          {/* preservedVideoUrl is always the untrimmed original (api.streamUrl
              serves the upload itself), so rally-only playback applies here
              too — see ResultsScreen for why previewKind can't gate this. */}
          <VideoPreview previewUrl={preservedVideoUrl} localFile={localFile}
            previewKind={previewKind} skipSegments={skipSegments} />
          {error && (
            <ProcessingError inline title="Analysis stopped" message={error}
              onRetry={retry} onHome={onHome} retryLabel="Retry analysis" />
          )}
        </section>
      </main>
      <Footer left="PROCESSING" right="PASS 3 OF 3" />
    </div>
  )
}
