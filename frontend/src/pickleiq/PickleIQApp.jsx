import React, { useCallback, useEffect, useRef, useState } from 'react'
import PrimaryButton from './components/PrimaryButton.jsx'
import SegmentedControl from './components/SegmentedControl.jsx'
import VideoUploadControl from './components/VideoUploadControl.jsx'
import Pass1Screen from './Pass1Screen.jsx'
import RallyReviewScreen from './RallyReviewScreen.jsx'
import Pass2Screen from './Pass2Screen.jsx'
import AnalysisScreen from './AnalysisScreen.jsx'
import ResultsScreen from './ResultsScreen.jsx'
import PastClipsList from './PastClipsList.jsx'
import SelfStopToggle from './components/SelfStopToggle.jsx'
import { loadClipResults, resumeReviewState, startPickleIQAnalysis } from './processingAdapter.js'

const GAME_OPTIONS = [
  { value: 'singles', label: 'Singles' },
  { value: 'doubles', label: 'Doubles' },
]
const ANALYSIS_OPTIONS = [
  { value: 'positioning', label: 'Positioning' },
  { value: 'form', label: 'Form' },
]

export default function PickleIQApp() {
  const [stage, setStage] = useState('upload')
  const [video, setVideo] = useState(null)
  const [videoRecord, setVideoRecord] = useState(null)
  const [gameType, setGameType] = useState('singles')
  const [analysisType, setAnalysisType] = useState('positioning')
  const [uploading, setUploading] = useState(false)
  const [pass1Output, setPass1Output] = useState(null)
  const [pass2Output, setPass2Output] = useState(null)
  const [analysisOutput, setAnalysisOutput] = useState(null)
  const [historyResult, setHistoryResult] = useState(null)
  const [historyError, setHistoryError] = useState('')
  const [error, setError] = useState('')
  const sessionRef = useRef(0)
  const uploadControllerRef = useRef(null)

  useEffect(() => () => {
    sessionRef.current += 1
    uploadControllerRef.current?.abort()
  }, [])

  const startAnalysis = async () => {
    if (!video || stage !== 'upload' || uploading) return
    setError('')
    setUploading(true)
    const session = sessionRef.current
    const controller = new AbortController()
    uploadControllerRef.current = controller
    try {
      const uploadedVideo = await startPickleIQAnalysis({
        file: video, gameType, analysisType, signal: controller.signal,
      })
      if (session !== sessionRef.current) return
      setVideoRecord(uploadedVideo)
      setStage('pass1')
    } catch (uploadError) {
      if (uploadError.name === 'AbortError' || session !== sessionRef.current) return
      setError(uploadError.message || 'We could not start the analysis. Please try again.')
    } finally {
      if (session === sessionRef.current) setUploading(false)
      if (uploadControllerRef.current === controller) uploadControllerRef.current = null
    }
  }

  const returnHome = () => {
    sessionRef.current += 1
    uploadControllerRef.current?.abort()
    uploadControllerRef.current = null
    setStage('upload')
    setVideo(null)
    setVideoRecord(null)
    setPass1Output(null)
    setPass2Output(null)
    setAnalysisOutput(null)
    setHistoryResult(null)
    setHistoryError('')
    setGameType('singles')
    setAnalysisType('positioning')
    setUploading(false)
    setError('')
  }

  const openHistory = () => setStage('past-clips')

  const selectPastClip = async (videoId, status) => {
    setHistoryError('')
    try {
      if (status === 'needs-review') {
        // Pass 1 already finished for this clip but nobody ever submitted
        // rally selections for it (interrupted tab, refresh, etc.) — jump
        // straight back into the same 'review' stage a live upload uses,
        // reusing its whole downstream chain (review -> pass2 -> analysis
        // -> results) unchanged. No local File object exists for a resumed
        // clip; resumeReviewState seeds a streamed previewUrl instead,
        // which RallyReviewScreen/Pass2Screen/AnalysisScreen already know
        // how to fall back to.
        const resumed = await resumeReviewState(videoId)
        if (!resumed) {
          setHistoryError('That clip has no detected rallies to review.')
          return
        }
        setVideoRecord({ id: videoId })
        setVideo(null)
        setPass1Output(resumed)
        setStage('review')
        return
      }
      const result = await loadClipResults(videoId)
      if (!result) {
        setHistoryError('That clip has not finished generating feedback yet.')
        return
      }
      setHistoryResult(result)
      setStage('history-results')
    } catch (loadError) {
      setHistoryError(loadError.message || 'Could not load that clip.')
    }
  }

  const finishPass1 = useCallback((output) => {
    setPass1Output(output)
    setStage('review')
  }, [])

  const finishReview = useCallback(() => {
    setStage('pass2')
  }, [])

  const finishPass2 = useCallback((output) => {
    setPass2Output(output)
    setStage('analysis')
  }, [])

  const finishAnalysis = useCallback((output) => {
    setAnalysisOutput({
      ...output,
      videoId: videoRecord.id,
      gameType,
      analysisType,
      keptSections: pass2Output.keptSections,
      rejectedSections: pass2Output.rejectedSections,
      videoSourceKind: pass2Output.previewKind,
    })
    setStage('results')
  }, [videoRecord, gameType, analysisType, pass2Output])

  if (stage === 'pass1') {
    return <Pass1Screen videoId={videoRecord.id} localFile={video}
      onComplete={finishPass1} onHome={returnHome} />
  }

  if (stage === 'review') {
    return <RallyReviewScreen videoId={videoRecord.id} localFile={video} pass1Result={pass1Output}
      onComplete={finishReview} onHome={returnHome} />
  }

  if (stage === 'pass2') {
    return <Pass2Screen videoId={videoRecord.id} localFile={video} pass1Result={pass1Output}
      onComplete={finishPass2} onHome={returnHome} />
  }

  if (stage === 'analysis') {
    return <AnalysisScreen videoId={videoRecord.id} localFile={video}
      pass2Result={pass2Output}
      finalVideoUrl={pass2Output?.finalVideoUrl || pass2Output?.previewUrl || null}
      onComplete={finishAnalysis} onHome={returnHome} />
  }

  if (stage === 'results') {
    return <ResultsScreen analysisResult={analysisOutput} pass2Result={pass2Output}
      localFile={video} onHome={returnHome} />
  }

  if (stage === 'past-clips') {
    return <PastClipsList onSelect={selectPastClip} onHome={returnHome} selectError={historyError} />
  }

  if (stage === 'history-results') {
    // ResultsScreen's button reads "Back to Home" — send it to the actual
    // home screen, not back into the list, so the label stays honest.
    return <ResultsScreen analysisResult={historyResult?.analysisResult} pass2Result={historyResult?.pass2Result}
      localFile={null} onHome={returnHome} />
  }

  return (
    <main className="pickleiq-page" aria-labelledby="pickleiq-title">
      <SelfStopToggle />
      <button type="button" className="pickleiq-secondary-button pickleiq-history-link" onClick={openHistory}>
        Past Clips
      </button>
      <h1 className="pickleiq-wordmark" id="pickleiq-title">PickleIQ</h1>
      <section className="pickleiq-control-stack" aria-label="Analysis setup">
        <VideoUploadControl file={video} disabled={uploading}
          onChange={(nextVideo) => { setVideo(nextVideo); setError('') }} onError={setError} />
        <SegmentedControl label="Game type" options={GAME_OPTIONS} value={gameType}
          onChange={setGameType} disabled={uploading} />
        <SegmentedControl label="Analysis type" options={ANALYSIS_OPTIONS} value={analysisType}
          onChange={setAnalysisType} tone={analysisType} disabled={uploading} />
      </section>
      {error && <p className="pickleiq-error" role="alert">{error}</p>}
      <PrimaryButton className="pickleiq-start-button" disabled={!video || uploading} onClick={startAnalysis}
        aria-busy={uploading}
        aria-describedby={!video ? 'pickleiq-start-help' : undefined}>
        {uploading ? 'Uploading…' : 'Start Analysis'}
      </PrimaryButton>
      <p id="pickleiq-start-help" className="pickleiq-visually-hidden">
        Choose a video before starting analysis.
      </p>
    </main>
  )
}
