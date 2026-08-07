import React, { useEffect, useRef, useState } from 'react'
import { resolveRallySkipTarget } from '../processingAdapter.js'
import { drawBall, drawPlayers, ensureChunks, entryAt, layoutCanvas, newCache } from '../../overlayDraw.js'

export default function VideoPreview({ previewUrl, localFile, previewKind, playerRef,
  skipSegments = null, overlayVideoId = null, overlayFps = 30 }) {
  const [localUrl, setLocalUrl] = useState(null)
  const videoUrl = previewUrl || localUrl
  const internalRef = useRef(null)
  const canvasRef = useRef(null)
  const skipSegmentsRef = useRef(skipSegments)
  skipSegmentsRef.current = skipSegments
  const [showOverlay, setShowOverlay] = useState(true)

  useEffect(() => {
    if (!localFile) {
      setLocalUrl(null)
      return undefined
    }
    const nextUrl = URL.createObjectURL(localFile)
    setLocalUrl(nextUrl)
    return () => URL.revokeObjectURL(nextUrl)
  }, [localFile])

  // Pass 2 detections drawn on a canvas over the original video — same stored
  // metadata the dev tool's Task 2 tab renders (see overlayDraw.js), minus the
  // debug layers. Only valid because every screen that passes overlayVideoId
  // plays the SOURCE timeline, so frame = currentTime * fps lines up.
  useEffect(() => {
    if (!overlayVideoId || !showOverlay || !videoUrl) return undefined
    const cache = newCache()
    const boxRef = { current: '' }
    let raf
    const tick = () => {
      raf = requestAnimationFrame(tick)
      const v = internalRef.current
      const cv = canvasRef.current
      if (!v || !cv || !v.videoWidth) return
      if (cv.width !== v.videoWidth) { cv.width = v.videoWidth; cv.height = v.videoHeight }
      layoutCanvas(v, cv, boxRef)
      const frame = Math.round(v.currentTime * (overlayFps || 30))
      ensureChunks(cache, overlayVideoId, 'pass2', frame)
      const g = cv.getContext('2d')
      g.clearRect(0, 0, cv.width, cv.height)
      const entry = entryAt(cache, frame)
      drawBall(g, cache, frame, entry, { events: false })
      drawPlayers(g, cache, frame, entry)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [overlayVideoId, overlayFps, showOverlay, videoUrl])

  const setRefs = (node) => {
    internalRef.current = node
    if (playerRef) playerRef.current = node
  }

  const onTimeUpdate = () => {
    const video = internalRef.current
    const segments = skipSegmentsRef.current
    if (!video || !segments?.length) return
    const target = resolveRallySkipTarget(segments, video.currentTime)
    if (target === 'end') {
      video.pause()
    } else if (typeof target === 'number' && target > video.currentTime) {
      video.currentTime = target
    }
  }

  const skipping = Boolean(skipSegments?.length)
  return (
    <figure className="pickleiq-video-preview">
      {videoUrl ? (
        <video ref={setRefs} src={videoUrl} controls preload="metadata"
          onTimeUpdate={onTimeUpdate} aria-label="Video preview" />
      ) : (
        <div className="pickleiq-video-placeholder" role="img" aria-label="Video preview unavailable" />
      )}
      {overlayVideoId && videoUrl && (
        <>
          {showOverlay && <canvas ref={canvasRef} className="pickleiq-video-overlay" />}
          <button type="button" className="pickleiq-video-overlay-toggle"
            aria-pressed={showOverlay}
            onClick={() => setShowOverlay((on) => !on)}>
            {showOverlay ? 'Hide tracking' : 'Show tracking'}
          </button>
        </>
      )}
      {skipping && videoUrl ? (
        <figcaption>Rally-only playback — dead time between rallies is skipped</figcaption>
      ) : previewKind === 'original-local' && videoUrl ? (
        <figcaption>Original video preview</figcaption>
      ) : null}
    </figure>
  )
}
