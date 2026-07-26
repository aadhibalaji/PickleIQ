import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'
import { buildPlayerColorMap, playerColor } from './playerColors.js'

// Court constants (metres) mirroring backend/app/cv/court.py — a 20x44ft
// pickleball court, net at mid-length, kitchen (NVZ) 7ft each side of net.
// y=0 is the FAR baseline (small y = far from camera), y=COURT_L the NEAR
// baseline — same convention refine/ballistic.py and cv/court.py use.
const COURT_W = 6.096
const COURT_L = 13.4112
const NET_Y = COURT_L / 2
const KITCHEN = 2.1336
const NET_HEIGHT_M = 0.9 // ~centre net height, just for the perspective cue

// Hand-rolled pseudo-3D projection: a trapezoidal floor (far baseline
// narrower, matching the axial "looking down the court" camera every
// backend module here assumes — see refine/ballistic.py, cv/bounce.py),
// with ball height lifting points up the screen. No 3D library — same
// hand-rolled-SVG approach as the rest of this UI.
const PAD = 20
const HEIGHT_PAD = 150 // headroom above the floor for lobs
const FLOOR_W_PX = 260 // near-baseline (widest) floor width in px
const FLOOR_DEPTH_PX = 300
const FAR_SCALE = 0.52 // far-baseline width as a fraction of the near baseline
const SCALE_X = FLOOR_W_PX / COURT_W
const HEIGHT_SCALE = 26 // px per metre of ball height

const SVG_W = FLOOR_W_PX + PAD * 2
const SVG_H = HEIGHT_PAD + FLOOR_DEPTH_PX + PAD

// Floor projection: returns [screenX, screenY, widthScaleAtThisDepth].
function projectFloor(x, y) {
  const t = Math.min(1, Math.max(0, y / COURT_L)) // 0 = far, 1 = near
  const wScale = FAR_SCALE + t * (1 - FAR_SCALE)
  const cx = PAD + FLOOR_W_PX / 2 + (x - COURT_W / 2) * SCALE_X * wScale
  const cy = HEIGHT_PAD + t * FLOOR_DEPTH_PX
  return [cx, cy, wScale]
}

// Full 3D point projection (x, y court metres; z ball height metres).
function project(x, y, z = 0) {
  const [cx, cyFloor, wScale] = projectFloor(x, y)
  return [cx, cyFloor - z * HEIGHT_SCALE * wScale]
}

function floorLine(y) {
  const [x0, cy] = project(0, y)
  const [x1] = project(COURT_W, y)
  return { x0, x1, y: cy }
}

export default function ShotHeatmap({ videoId }) {
  const [data, setData] = useState(null)
  const [showServes, setShowServes] = useState(true)

  useEffect(() => {
    setData(null)
    api.getTrajectories(videoId).then((r) => setData(r.rallies || [])).catch(() => setData([]))
  }, [videoId])

  // Flatten per-rally points into arcs: a flight segment is already emitted
  // frame-contiguous (refine/ballistic.py fits one parabola per unbroken
  // tracked run), so a break in frame continuity marks a new arc.
  const arcs = useMemo(() => {
    if (!data) return []
    const out = []
    for (const rally of data) {
      let cur = []
      for (const p of rally.points || []) {
        if (cur.length && p.frame - cur[cur.length - 1].frame !== 1) {
          out.push(cur)
          cur = []
        }
        cur.push(p)
      }
      if (cur.length) out.push(cur)
    }
    return out
  }, [data])

  if (data === null) return <div className="clip-list-empty">Loading trajectory map…</div>

  const visibleArcs = arcs.filter((a) => showServes || !a[0].is_serve)
  const colorMap = buildPlayerColorMap(arcs.map((a) => a[0].player_id))
  const playerIds = Object.keys(colorMap).map(Number).sort((a, b) => a - b)

  const farKitchen = floorLine(NET_Y - KITCHEN)
  const nearKitchen = floorLine(NET_Y + KITCHEN)
  const netFloor = floorLine(NET_Y)
  const netTopL = project(0, NET_Y, NET_HEIGHT_M)
  const netTopR = project(COURT_W, NET_Y, NET_HEIGHT_M)
  const centerFar = project(COURT_W / 2, 0)
  const centerNear = project(COURT_W / 2, COURT_L)
  const corners = [project(0, 0), project(COURT_W, 0), project(COURT_W, COURT_L), project(0, COURT_L)]

  return (
    <div>
      <div className="row spread">
        <h2>3D Trajectory Map</h2>
        <label className="row shot-heatmap-toggle" style={{ fontSize: 12, color: 'var(--muted)' }}>
          <input type="checkbox" checked={showServes} onChange={(e) => setShowServes(e.target.checked)} />
          Show serves
        </label>
      </div>

      {visibleArcs.length === 0 ? (
        <div className="clip-list-empty">No reconstructed ball trajectories yet.</div>
      ) : (
        <div className="shot-heatmap">
          <svg width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`}>
            {/* trapezoidal floor */}
            <polygon points={corners.map(([x, y]) => `${x},${y}`).join(' ')}
                     fill="none" stroke="var(--border)" strokeWidth="2" />
            <line x1={farKitchen.x0} y1={farKitchen.y} x2={farKitchen.x1} y2={farKitchen.y}
                  stroke="var(--border)" strokeWidth="1" strokeDasharray="4 3" />
            <line x1={nearKitchen.x0} y1={nearKitchen.y} x2={nearKitchen.x1} y2={nearKitchen.y}
                  stroke="var(--border)" strokeWidth="1" strokeDasharray="4 3" />
            <line x1={centerFar[0]} y1={centerFar[1]} x2={centerNear[0]} y2={centerNear[1]}
                  stroke="var(--border)" strokeWidth="1" strokeDasharray="2 4" />
            {/* net: floor line + lifted top line + posts, for the depth cue */}
            <line x1={netFloor.x0} y1={netFloor.y} x2={netFloor.x1} y2={netFloor.y}
                  stroke="var(--text)" strokeWidth="2" />
            <line x1={netTopL[0]} y1={netTopL[1]} x2={netTopR[0]} y2={netTopR[1]}
                  stroke="var(--text)" strokeWidth="1.5" opacity="0.7" />
            <line x1={netFloor.x0} y1={netFloor.y} x2={netTopL[0]} y2={netTopL[1]}
                  stroke="var(--text)" strokeWidth="1" opacity="0.7" />
            <line x1={netFloor.x1} y1={netFloor.y} x2={netTopR[0]} y2={netTopR[1]}
                  stroke="var(--text)" strokeWidth="1" opacity="0.7" />

            {/* trajectory arcs: color = attributed player, opacity = confidence
                (two independent channels — never conflated) */}
            {visibleArcs.map((arc, ai) => {
              const color = playerColor(colorMap, arc[0].player_id)
              return (
                <g key={ai}>
                  {arc.slice(1).map((p, i) => {
                    const prev = arc[i]
                    const [x0, y0] = project(prev.x, prev.y, prev.height_m)
                    const [x1, y1] = project(p.x, p.y, p.height_m)
                    const segConf = (prev.confidence + p.confidence) / 2
                    return (
                      <line key={i} x1={x0} y1={y0} x2={x1} y2={y1}
                            stroke={color} strokeWidth="2.5" opacity={Math.max(0.12, segConf)} />
                    )
                  })}
                  {arc.map((p, i) => {
                    const [x, y] = project(p.x, p.y, p.height_m)
                    const title = `Player ${p.player_id ?? '?'}${p.is_serve ? ' · serve' : ''} · frame ${p.frame} · h=${p.height_m.toFixed(2)}m · conf=${p.confidence.toFixed(2)}`
                    return (
                      <circle key={i} cx={x} cy={y} r={p.is_serve ? 3.5 : 2.5}
                              fill={color} opacity={Math.max(0.15, p.confidence)}>
                        <title>{title}</title>
                      </circle>
                    )
                  })}
                </g>
              )
            })}
          </svg>

          <div className="shot-heatmap-legend">
            {playerIds.map((pid) => (
              <span className="row" key={pid}>
                <span className="shot-heatmap-swatch" style={{ background: playerColor(colorMap, pid) }} /> Player {pid}
              </span>
            ))}
            <span className="row"><span className="shot-heatmap-swatch" style={{ background: 'var(--muted)' }} /> Unattributed</span>
            <span className="muted" style={{ maxWidth: 220 }}>
              Faded = low confidence (poor arc fit and/or far-court depth noise)
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
