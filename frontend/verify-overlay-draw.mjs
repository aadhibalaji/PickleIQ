import assert from 'node:assert/strict'
import { drawBall, drawPlayers, entryAt, layoutCanvas } from './src/overlayDraw.js'

// Minimal recording stand-in for a CanvasRenderingContext2D.
const fakeCtx = () => {
  const calls = []
  const ctx = { globalAlpha: 1 }
  for (const m of ['beginPath', 'moveTo', 'lineTo', 'stroke', 'fill', 'arc',
                   'strokeRect', 'fillText', 'clearRect']) {
    ctx[m] = (...args) => calls.push([m, ...args])
  }
  ctx.calls = calls
  return ctx
}
const count = (ctx, m) => ctx.calls.filter((c) => c[0] === m).length

const cache = { fetched: new Set(), byFrame: new Map() }
const kp = Array.from({ length: 17 }, () => [50, 50])
cache.byFrame.set(100, { f: 100, players: [{ box: [1, 2, 11, 22], team: 'near', num: 1, kp }] })
cache.byFrame.set(140, { f: 140, ball: [30, 40, 8] })
cache.byFrame.set(142, { f: 142, ball: [35, 45, 8], evt: { kind: 'bounce', in: true } })

// entryAt looks back up to 8 frames (pass1 is subsampled), no further.
assert.equal(entryAt(cache, 100)?.f, 100)
assert.equal(entryAt(cache, 108)?.f, 100)
assert.equal(entryAt(cache, 109), null)

// Players fall back up to 90 frames and render faded when stale.
let g = fakeCtx()
drawPlayers(g, cache, 100, entryAt(cache, 100))
assert.equal(count(g, 'strokeRect'), 1)
assert.deepEqual(g.calls.find((c) => c[0] === 'fillText').slice(1, 2), ['P1'])
assert.equal(g.globalAlpha, 1)

g = fakeCtx()
drawPlayers(g, cache, 189, entryAt(cache, 189))   // stale fallback, still drawn
assert.equal(count(g, 'strokeRect'), 1)
g = fakeCtx()
drawPlayers(g, cache, 191, entryAt(cache, 191))   // beyond the 90-frame window
assert.equal(count(g, 'strokeRect'), 0)

// Pose skeleton: one line per edge, one joint dot per detected keypoint.
g = fakeCtx()
drawPlayers(g, cache, 100, entryAt(cache, 100))
assert.equal(count(g, 'lineTo'), 14)
assert.equal(count(g, 'fill'), 17)
g = fakeCtx()
drawPlayers(g, cache, 100, { f: 100, players: [{ box: [1, 2, 11, 22], team: 'far' }] })
assert.equal(count(g, 'lineTo'), 0)   // no keypoints -> box only, no crash

// Ball: trail covers the last 24 frames; the event marker adds its label.
g = fakeCtx()
drawBall(g, cache, 145, entryAt(cache, 145))
assert.equal(count(g, 'arc'), 3)                  // two trail dots + one event ring
assert.deepEqual(g.calls.find((c) => c[0] === 'fillText').slice(1, 2), ['bounce'])
g = fakeCtx()
drawBall(g, cache, 200, entryAt(cache, 200))      // trail window passed
assert.equal(count(g, 'arc'), 0)

// events:false (user-side overlay) keeps the trail, drops the ring and label.
g = fakeCtx()
drawBall(g, cache, 145, entryAt(cache, 145), { events: false })
assert.equal(count(g, 'arc'), 2)
assert.equal(count(g, 'fillText'), 0)

// layoutCanvas matches the video's letterboxed content box, not the element.
const video = { videoWidth: 1920, videoHeight: 1080, clientWidth: 800, clientHeight: 600,
                offsetLeft: 0, offsetTop: 0 }
const canvas = { style: {} }
const boxRef = { current: '' }
layoutCanvas(video, canvas, boxRef)
assert.deepEqual(canvas.style, { left: '0px', top: '75px', width: '800px', height: '450px' })
canvas.style = {}
layoutCanvas(video, canvas, boxRef)                // unchanged box -> no reflow
assert.deepEqual(canvas.style, {})

console.log('Overlay draw fixtures passed')
