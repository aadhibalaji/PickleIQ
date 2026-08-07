// Overlay chunk cache + canvas drawing shared by the dev tool's VideoPlayer
// and the user-facing pickleiq VideoPreview. Both draw stored pass1/pass2
// metadata onto a canvas over the ORIGINAL video — no video file is edited.
import { api } from './api.js'

export const CHUNK = 900 // must match backend overlays.CHUNK_FRAMES

// COCO-17 skeleton edges (YOLO11-pose keypoint indices)
const SKELETON = [
  [5, 7], [7, 9], [6, 8], [8, 10],          // arms
  [5, 6], [5, 11], [6, 12], [11, 12],       // torso
  [11, 13], [13, 15], [12, 14], [14, 16],   // legs
  [0, 5], [0, 6],                           // head to shoulders
]

export const newCache = () => ({ fetched: new Set(), byFrame: new Map() })

export function ensureChunks(c, videoId, source, frame) {
  const ci = Math.floor(frame / CHUNK)
  for (const idx of [ci, ci + 1]) {
    if (c.fetched.has(idx)) continue
    c.fetched.add(idx)
    api.getOverlays(videoId, source, idx * CHUNK, (idx + 1) * CHUNK - 1)
      .then((r) => r.entries.forEach((e) => c.byFrame.set(e.f, e)))
      .catch(() => c.fetched.delete(idx))
  }
}

// Nearest entry at or just before the current frame (pass1 is subsampled).
export function entryAt(c, frame) {
  for (let d = 0; d <= 8; d++) {
    const e = c.byFrame.get(frame - d)
    if (e) return e
  }
  return null
}

// Size/position the overlay canvas to EXACTLY cover the video's rendered
// (letterboxed) content box. Everything is drawn on this one canvas, so this
// is what keeps it aligned with the video in EVERY mode — especially
// fullscreen, where the video is object-fit:contain and the wrapper is larger
// than the video, so a canvas that merely filled the wrapper would stretch the
// overlay (visible first on the long, precise court lines).
export function layoutCanvas(v, cv, boxRef) {
  if (!v || !cv || !v.videoWidth || !v.clientWidth) return
  const scale = Math.min(v.clientWidth / v.videoWidth, v.clientHeight / v.videoHeight)
  const dw = Math.round(v.videoWidth * scale)
  const dh = Math.round(v.videoHeight * scale)
  const left = Math.round(v.offsetLeft + (v.clientWidth - dw) / 2)
  const top = Math.round(v.offsetTop + (v.clientHeight - dh) / 2)
  const box = `${left},${top},${dw},${dh}`
  if (box === boxRef.current) return   // avoid reflow when unchanged
  boxRef.current = box
  cv.style.left = `${left}px`
  cv.style.top = `${top}px`
  cv.style.width = `${dw}px`
  cv.style.height = `${dh}px`
}

// Ball trail from recent entries — red so it contrasts with the (often yellow)
// pickleball itself; current position gets a white outline. `events` adds the
// bounce/hit ring + label, which linger a few frames so they're visible at
// speed — dev-tool only; the user-facing overlay shows the bare trail.
export function drawBall(g, c, frame, entry, { events = true } = {}) {
  for (let d = 24; d >= 0; d--) {
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
    if (events && e?.evt && frame - e.f <= 45) {
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
}

// Players: pass2 has them every frame; pass1 only at sparse spot-checks, so
// fall back to the most recent entry that carried players (faded).
export function drawPlayers(g, c, frame, entry) {
  let pentry = entry
  let stale = false
  if (!pentry?.players) {
    for (let d = 1; d <= 90; d++) {
      const e = c.byFrame.get(frame - d)
      if (e?.players) { pentry = e; stale = true; break }
    }
  }
  if (!pentry?.players) return
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
