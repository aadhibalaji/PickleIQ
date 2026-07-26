import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { buildPlayerColorMap, playerColor } from './playerColors.js'

// Court constants (metres) mirroring backend/app/cv/court.py — a 20x44ft
// pickleball court, net at mid-length, kitchen (NVZ) 7ft each side of net.
const COURT_W = 6.096
const COURT_L = 13.4112
const NET_Y = COURT_L / 2
const KITCHEN = 2.1336

const PAD = 20
const SCALE = 40 // px per metre
const SVG_W = COURT_W * SCALE + PAD * 2
const SVG_H = COURT_L * SCALE + PAD * 2

const toSvgX = (x) => PAD + x * SCALE
const toSvgY = (y) => PAD + y * SCALE

export default function ShotHeatmap({ videoId }) {
  const [events, setEvents] = useState(null)
  const [showServes, setShowServes] = useState(true)

  useEffect(() => {
    setEvents(null)
    api.getContactEvents(videoId).then((r) => setEvents(r.events || [])).catch(() => setEvents([]))
  }, [videoId])

  if (events === null) return <div className="clip-list-empty">Loading shot map…</div>

  // Map raw rows to plot points. `height_m` is a forward-compat slot for a
  // future 3D/lob-height upgrade — always null today, no data source yet.
  const points = events
    .filter((e) => e.court_x != null && e.court_y != null)
    .filter((e) => showServes || !e.is_serve)
    .map((e) => ({
      contactId: e.contact_id,
      x: e.court_x,
      y: e.court_y,
      height_m: null, // ponytail: forward-compat slot, wire up when 3D contact height lands
      playerId: e.player_id,
      team: e.team,
      isServe: !!e.is_serve,
      shotSide: e.shot_side,
      timestampS: e.timestamp_s,
      confidence: e.confidence,
    }))

  // Same player1/player2 mapping rule used across the panel (tips, filter
  // tabs, report stats): sort distinct player_ids ascending, first = token 1.
  // Keyed by player_id (not team) so it stays consistent with the rest of the
  // Rally Analysis panel, which only knows player_id/player_label.
  const colorMap = buildPlayerColorMap(points.map((p) => p.playerId))
  const playerIds = Object.keys(colorMap).map(Number).sort((a, b) => a - b)

  return (
    <div>
      <div className="row spread">
        <h2>Shot Map</h2>
        <label className="row shot-heatmap-toggle" style={{ fontSize: 12, color: 'var(--muted)' }}>
          <input type="checkbox" checked={showServes} onChange={(e) => setShowServes(e.target.checked)} />
          Show serves
        </label>
      </div>

      {points.length === 0 ? (
        <div className="clip-list-empty">No contact events yet.</div>
      ) : (
        <div className="shot-heatmap">
          <svg width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`}>
            <rect x={PAD} y={PAD} width={COURT_W * SCALE} height={COURT_L * SCALE}
                  fill="none" stroke="var(--border)" strokeWidth="2" />
            <line x1={PAD} y1={toSvgY(NET_Y)} x2={PAD + COURT_W * SCALE} y2={toSvgY(NET_Y)}
                  stroke="var(--text)" strokeWidth="2" />
            <line x1={PAD} y1={toSvgY(NET_Y - KITCHEN)} x2={PAD + COURT_W * SCALE} y2={toSvgY(NET_Y - KITCHEN)}
                  stroke="var(--border)" strokeWidth="1" strokeDasharray="4 3" />
            <line x1={PAD} y1={toSvgY(NET_Y + KITCHEN)} x2={PAD + COURT_W * SCALE} y2={toSvgY(NET_Y + KITCHEN)}
                  stroke="var(--border)" strokeWidth="1" strokeDasharray="4 3" />
            <line x1={PAD + (COURT_W / 2) * SCALE} y1={PAD} x2={PAD + (COURT_W / 2) * SCALE} y2={PAD + COURT_L * SCALE}
                  stroke="var(--border)" strokeWidth="1" strokeDasharray="2 4" />

            {points.map((p) => {
              const color = playerColor(colorMap, p.playerId)
              const title = `Player ${p.playerId} (${p.team}) · ${p.shotSide}${p.isServe ? ' · serve' : ''} · ${formatTime(p.timestampS)}`
              return p.isServe ? (
                <circle key={p.contactId} cx={toSvgX(p.x)} cy={toSvgY(p.y)} r={5}
                        fill="none" stroke={color} strokeWidth="2">
                  <title>{title}</title>
                </circle>
              ) : (
                <circle key={p.contactId} cx={toSvgX(p.x)} cy={toSvgY(p.y)} r={4}
                        fill={color} opacity="0.85">
                  <title>{title}</title>
                </circle>
              )
            })}
          </svg>

          <div className="shot-heatmap-legend">
            {playerIds.map((pid) => (
              <span className="row" key={pid}>
                <span className="shot-heatmap-swatch" style={{ background: playerColor(colorMap, pid) }} /> Player {pid}
              </span>
            ))}
            <span className="row"><span className="shot-heatmap-serve-swatch" /> Serve</span>
          </div>
        </div>
      )}
    </div>
  )
}

function formatTime(s) {
  if (s == null) return ''
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}
