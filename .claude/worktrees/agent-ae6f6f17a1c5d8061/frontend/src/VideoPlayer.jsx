import React, { useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react'
import { api } from './api.js'

const CHUNK = 900 // must match backend overlays.CHUNK_FRAMES

// COCO-17 skeleton edges (YOLO11-pose keypoint indices)
const SKELETON = [
  [5, 7], [7, 9], [6, 8], [8, 10],          // arms
  [5, 6], [5, 11], [6, 12], [11, 12],       // torso
  [11, 13], [13, 15], [12, 14], [14, 16],   // legs
  [0, 5], [0, 6],                           // head to shoulders
]

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

  const ensureChunks = (source, frame) => {
    if (!source) return
    let c = cacheRef.current[source]
    if (!c) c = cacheRef.current[source] = { fetched: new Set(), byFrame: new Map() }
    const ci = Math.floor(frame / CHUNK)
    for (const idx of [ci, ci + 1]) {
      if (c.fetched.has(idx)) continue
      c.fetched.add(idx)
      api.getOverlays(videoId, source, idx * CHUNK, (idx + 1) * CHUNK - 1)
        .then((r) => r.entries.forEach((e) => c.byFrame.set(e.f, e)))
        .catch(() => c.fetched.delete(idx))
    }
  }

  // Size/position the overlay canvas to EXACTLY cover the video's rendered
  // (letterboxed) content box. Court lines + player boxes are all drawn on this
  // one canvas, so this is what keeps them aligned with the video in EVERY mode
  // — especially fullscreen, where the video is object-fit:contain and the
  // wrapper is larger than the video, so a canvas that merely filled the wrapper
  // would stretch the overlay (visible first on the long, precise court lines).
  const layoutCanvas = () => {
    const v = videoRef.current
    const cv = canvasRef.current
    if (!v || !cv || !v.videoWidth || !v.clientWidth) return
    const scale = Math.min(v.clientWidth / v.videoWidth, v.clientHeight / v.videoHeight)
    const dw = Math.round(v.videoWidth * scale)
    const dh = Math.round(v.videoHeight * scale)
    const left = Math.round(v.offsetLeft + (v.clientWidth - dw) / 2)
    const top = Math.round(v.offsetTop + (v.clientHeight - dh) / 2)
    const box = `${left},${top},${dw},${dh}`
    if (box === lastBoxRef.current) return   // avoid reflow when unchanged
    lastBoxRef.current = box
    cv.style.left = `${left}px`
    cv.style.top = `${top}px`
    cv.style.width = `${dw}px`
    cv.style.height = `${dh}px`
  }

  const draw = () => {
    const v = videoRef.current
    const cv = canvasRef.current
    if (!v || !cv || !v.videoWidth) return
    if (cv.width !== v.videoWidth) { cv.width = v.videoWidth; cv.height = v.videoHeight }
    layoutCanvas()
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

    // Nearest entry at or just before the current frame (pass1 is subsampled).
    let entry = null
    for (let d = 0; d <= 8; d++) {
      entry = c.byFrame.get(frame - d)
      if (entry) break
    }

    // Ball trail from recent entries — red so it contrasts with the (often
    // yellow) pickleball itself; current position gets a white outline.
    // Toggleable from the player controls (🎾).
    if (stateRef.current.showBall) for (let d = 24; d >= 0; d--) {
      const e = c.byFrame.get(frame - d)
      if (e?.ball) {
        const a = 1 - d / 28
        const isNow = d === 0 || (entry && e.f === entry.f)
        g.fillStyle = isNow ? 'rgba(255,45,60,0.95)' : `rgba(255,45,60,${0.4 * a})`
        g.beginPath()
        g.arc(e.ball[0], e.ball[1], Math.max(4, (e.ball[2] || 7) / 2 + 1), 0, Math.PI * 2)
        g.fill()
        if (isNow) {
          g.strokeStyle = 'rgba(255,255,255,0.9)'
          g.lineWidth = 2
          g.stroke()
        }
      }
      // Event markers linger a few frames so they're visible at speed.
      if (e?.evt && frame - e.f <= 45) {
        const { evt } = e
        g.strokeStyle = evt.kind === 'bounce'
          ? (evt.in ? 'rgba(0,229,255,0.95)' : 'rgba(255,255,255,0.95)')
          : 'rgba(229,165,58,0.95)'
        g.lineWidth = 4
        if (e.ball) {
          g.beginPath(); g.arc(e.ball[0], e.ball[1], 16, 0, Math.PI * 2); g.stroke()
          g.fillStyle = g.strokeStyle
          g.font = 'bold 17px system-ui'
          g.fillText(evt.kind === 'bounce' ? (evt.in ? 'bounce' : 'bounce OUT') : 'hit',
                     e.ball[0] + 20, e.ball[1] - 10)
        }
      }
    }

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

    // Players: pass2 has them every frame; pass1 only at sparse spot-checks,
    // so fall back to the most recent entry that carried players (faded).
    let pentry = entry
    let stale = false
    if (!pentry?.players) {
      for (let d = 1; d <= 90; d++) {
        const e = c.byFrame.get(frame - d)
        if (e?.players) { pentry = e; stale = true; break }
      }
    }
    if (pentry?.players) {
      g.globalAlpha = stale ? 0.45 : 1.0
      g.font = 'bold 14px system-ui'
      for (const p of pentry.players) {
        const [x1, y1, x2, y2] = p.box
        const color = p.team === 'near' ? 'rgba(88,166,255,0.9)' : 'rgba(240,136,62,0.9)'
        g.strokeStyle = color
        g.lineWidth = 3
        g.strokeRect(x1, y1, x2 - x1, y2 - y1)
        g.fillStyle = color
        g.fillText(p.num != null ? `P${p.num}` : `#${p.id}`, x1 + 3, y1 - 5)
        // Pose skeleton (YOLO11-pose keypoints; [0,0] = joint not detected)
        if (p.kp?.length >= 17) {
          const ok = (i) => p.kp[i][0] > 1 && p.kp[i][1] > 1
          g.lineWidth = 3
          for (const [a, b] of SKELETON) {
            if (ok(a) && ok(b)) {
              g.beginPath()
              g.moveTo(p.kp[a][0], p.kp[a][1])
              g.lineTo(p.kp[b][0], p.kp[b][1])
              g.stroke()
            }
          }
          for (let i = 0; i < 17; i++) {
            if (ok(i)) {
              g.beginPath()
              g.arc(p.kp[i][0], p.kp[i][1], 3, 0, Math.PI * 2)
              g.fill()
            }
          }
        }
      }
      g.globalAlpha = 1.0
    }
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
      ensureChunks(stateRef.current.overlaySource, f)
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
