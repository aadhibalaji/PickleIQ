import React, { useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react'
import { api } from './api.js'
import { drawBall, drawPlayers, ensureChunks, entryAt, layoutCanvas, newCache } from './overlayDraw.js'

// Renders the ORIGINAL video with all annotation drawn from stored metadata
// onto a canvas overlay — the video file is never edited.
const VideoPlayer = forwardRef(function VideoPlayer(
  { videoId, meta, overlaySource, courtLines, clickMode, onCanvasClick, onFrame, overlayNonce,
    gtMarks, pendingMark, selectedMarkId },
  ref,
) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const lastBoxRef = useRef('')
  const [isFs, setIsFs] = useState(false)
  const [showVideo, setShowVideo] = useState(true)
  const [showBall, setShowBall] = useState(true)
  const cacheRef = useRef({}) // source -> { fetched:Set(chunk), byFrame:Map }
  const stateRef = useRef({ frame: 0, overlaySource, courtLines })
  stateRef.current.overlaySource = overlaySource
  stateRef.current.courtLines = courtLines
  stateRef.current.exclusionZones = meta.exclusion_zones
  stateRef.current.gtMarks = gtMarks
  stateRef.current.pendingMark = pendingMark
  stateRef.current.selectedMarkId = selectedMarkId
  stateRef.current.showBall = showBall

  useImperativeHandle(ref, () => ({
    seekFrame: (f) => {
      const v = videoRef.current
      if (v && meta.fps) v.currentTime = f / meta.fps + 1e-4
    },
    currentFrame: () => stateRef.current.frame,
    video: () => videoRef.current,
  }))

  const fetchChunks = (source, frame) => {
    if (!source) return
    let c = cacheRef.current[source]
    if (!c) c = cacheRef.current[source] = newCache()
    ensureChunks(c, videoId, source, frame)
  }

  const draw = () => {
    const v = videoRef.current
    const cv = canvasRef.current
    if (!v || !cv || !v.videoWidth) return
    if (cv.width !== v.videoWidth) { cv.width = v.videoWidth; cv.height = v.videoHeight }
    layoutCanvas(v, cv, lastBoxRef)
    const g = cv.getContext('2d')
    g.clearRect(0, 0, cv.width, cv.height)
    const { overlaySource: src, courtLines: lines } = stateRef.current
    const frame = stateRef.current.frame

    if (lines) {
      g.strokeStyle = 'rgba(76,195,138,0.85)'
      g.lineWidth = 3
      for (const [[x1, y1], [x2, y2]] of lines) {
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke()
      }
    }
    if (!src) return

    // Player-detection exclusion zones (normalized polygons from the fixed
    // camera): detections whose box centre lands here are dropped by the
    // backend. Drawn semi-transparent red + outline so they're visible on the
    // video AND in black-background mode.
    const zones = stateRef.current.exclusionZones
    if (zones) {
      for (const poly of zones) {
        if (!poly?.length) continue
        g.beginPath()
        poly.forEach(([nx, ny], i) => {
          const x = nx * cv.width, y = ny * cv.height
          if (i === 0) g.moveTo(x, y); else g.lineTo(x, y)
        })
        g.closePath()
        g.fillStyle = 'rgba(229,45,45,0.22)'
        g.fill()
        g.strokeStyle = 'rgba(229,45,45,0.9)'
        g.lineWidth = 3
        g.stroke()
      }
    }

    const c = cacheRef.current[src]
    if (!c) return

    const entry = entryAt(c, frame)

    // Ball trail + bounce/hit markers. Toggleable from the controls (🎾).
    if (stateRef.current.showBall) drawBall(g, c, frame, entry)

    // Ground-truth ball marks (marking mode): diamonds at their exact frame,
    // faded within +/-2 frames so a nearby mark is visible while stepping.
    const marks = stateRef.current.gtMarks
    if (marks?.length) {
      for (const m of marks) {
        const d = Math.abs(frame - m.frame)
        if (d > 2) continue
        const color = m.event_type === 'bounce'
          ? (m.in_out === 'out' ? 'rgba(255,84,112,1)' : 'rgba(0,229,255,1)')
          : m.event_type === 'hit' ? 'rgba(229,165,58,1)' : 'rgba(195,206,74,1)'
        g.globalAlpha = d === 0 ? 1.0 : 0.35
        g.strokeStyle = color
        g.lineWidth = 3
        const r = 11
        g.beginPath()
        g.moveTo(m.x, m.y - r); g.lineTo(m.x + r, m.y)
        g.lineTo(m.x, m.y + r); g.lineTo(m.x - r, m.y)
        g.closePath(); g.stroke()
        if (d === 0) {
          g.fillStyle = color
          g.font = 'bold 13px system-ui'
          const label = m.event_type === 'bounce'
            ? `B ${m.in_out || ''}${m.disagree ? ' ⚠' : ''}`
            : m.event_type === 'hit' ? 'H' : '·'
          g.fillText(label, m.x + r + 4, m.y - 4)
        }
        if (m.id === stateRef.current.selectedMarkId) {
          g.strokeStyle = 'rgba(255,255,255,0.95)'
          g.lineWidth = 2
          g.beginPath(); g.arc(m.x, m.y, r + 5, 0, Math.PI * 2); g.stroke()
        }
        g.globalAlpha = 1.0
      }
    }
    // Pending (unsaved) mark: yellow crosshair where the user just clicked.
    const pm = stateRef.current.pendingMark
    if (pm) {
      g.strokeStyle = 'rgba(255,221,64,0.95)'
      g.lineWidth = 2
      g.beginPath(); g.moveTo(pm.x - 14, pm.y); g.lineTo(pm.x + 14, pm.y); g.stroke()
      g.beginPath(); g.moveTo(pm.x, pm.y - 14); g.lineTo(pm.x, pm.y + 14); g.stroke()
      g.beginPath(); g.arc(pm.x, pm.y, 7, 0, Math.PI * 2); g.stroke()
    }

    drawPlayers(g, c, frame, entry)
    // Reconstructed ball height above the court plane (Pass 2 only).
    if (stateRef.current.showBall && entry?.world && entry.world[2] != null && entry.ball) {
      g.fillStyle = 'rgba(255,255,255,0.95)'
      g.font = 'bold 13px system-ui'
      g.fillText(`h ${entry.world[2].toFixed(2)}m`, entry.ball[0] + 14, entry.ball[1] + 16)
    }
    if (entry?.me != null) {
      g.fillStyle = 'rgba(147,164,181,0.9)'
      g.font = '12px system-ui'
      g.fillText(`motion ${entry.me}`, 10, cv.height - 12)
    }
  }

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    let raf
    const tick = () => {
      const f = Math.round(v.currentTime * (meta.fps || 30))
      if (f !== stateRef.current.frame) {
        stateRef.current.frame = f
        onFrame?.(f)
      }
      fetchChunks(stateRef.current.overlaySource, f)
      draw()
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoId, meta.fps])

  // Invalidate the cache when the source's results are regenerated.
  useEffect(() => { cacheRef.current = {} }, [videoId])

  // Pass 2 writes overlays one window at a time as it runs, so a chunk fetched
  // before its window finished was cached EMPTY and would otherwise never
  // refetch (symptom: players show for the first rally but not later ones while
  // processing). When overlayNonce bumps (parent polls it while a job is active),
  // drop the 'fetched' marks so chunks refetch; keep byFrame so nothing flickers.
  useEffect(() => {
    for (const src in cacheRef.current) cacheRef.current[src].fetched.clear()
  }, [overlayNonce])

  // Fullscreen the WRAPPER (video + canvas together), not the raw <video> —
  // otherwise the browser's native fullscreen shows only the video and the
  // overlay canvas (a sibling) disappears. The native fullscreen button is
  // hidden via controlsList; this is the single fullscreen entry point.
  useEffect(() => {
    const onFs = () => setIsFs(document.fullscreenElement === wrapRef.current)
    document.addEventListener('fullscreenchange', onFs)
    return () => document.removeEventListener('fullscreenchange', onFs)
  }, [])
  const toggleFullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen?.()
    else wrapRef.current?.requestFullscreen?.()
  }

  const handleClick = (e) => {
    if (clickMode && onCanvasClick) {
      const cv = canvasRef.current
      const rect = cv.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / rect.width) * cv.width
      const y = ((e.clientY - rect.top) / rect.height) * cv.height
      onCanvasClick(x, y)
      return
    }
    // Black-background mode hides the native <video> controls, so a click on the
    // overlay is the only way to pause/play — toggle it here.
    if (!showVideo) {
      const v = videoRef.current
      if (v) v.paused ? v.play() : v.pause()
    }
  }

  return (
    <div ref={wrapRef}
         className={`videowrap${clickMode ? ' clickable' : ''}${showVideo ? '' : ' novideo'}`}>
      <video ref={videoRef} src={api.streamUrl(videoId)} controls
             controlsList="nofullscreen" preload="metadata" />
      <canvas ref={canvasRef} onClick={handleClick} />
      <div className="vpbtns">
        <button type="button" className="fsbtn" onClick={() => setShowBall((s) => !s)}
                title={showBall ? 'Hide ball tracking (trail, bounce/hit markers, height)'
                                : 'Show ball tracking (trail, bounce/hit markers, height)'}
                style={showBall ? undefined : { opacity: 0.45 }}>
          🎾
        </button>
        <button type="button" className="fsbtn" onClick={() => setShowVideo((s) => !s)}
                title={showVideo ? 'Hide the video — show only the overlay on black'
                                 : 'Show the video behind the overlay'}>
          {showVideo ? '🎥' : '⬛'}
        </button>
        <button type="button" className="fsbtn" onClick={toggleFullscreen}
                title={isFs ? 'Exit fullscreen' : 'Fullscreen (keeps the overlay)'}>
          {isFs ? '✕' : '⛶'}
        </button>
      </div>
    </div>
  )
})

export default VideoPlayer
