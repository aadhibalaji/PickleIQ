import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import ClipList from './ClipList.jsx'
import DominanceBar from './DominanceBar.jsx'
import StatBlock from './StatBlock.jsx'

export default function RallyAnalysisPanel({ videoId, onSeek }) {
  const [position, setPosition] = useState(null)
  useEffect(() => {
    api.getPositionMetrics(videoId).then(setPosition).catch(() => setPosition(null))
  }, [videoId])

  const [shots, setShots] = useState(null)
  useEffect(() => {
    api.getShotMetrics(videoId).then(setShots).catch(() => setShots(null))
  }, [videoId])

  const [highlightClips, setHighlightClips] = useState([])
  useEffect(() => {
    api.getRallyHighlights(videoId)
      .then((r) => setHighlightClips((r.highlights || []).map((h) => ({
        label: `${h.tag} — ${h.rally_id}`, timestampS: h.timestamp_s, onSeek,
      }))))
      .catch(() => setHighlightClips([]))
  }, [videoId, onSeek])

  const [report, setReport] = useState(null)
  const [generating, setGenerating] = useState(false)
  const loadReport = () =>
    api.getCoachingReport(videoId).then((r) => setReport(r.report)).catch(() => setReport(null))
  useEffect(() => { loadReport() }, [videoId])

  const onGenerateReport = () => {
    setGenerating(true)
    api.generateCoachingReport(videoId).then(setReport).catch(() => {}).finally(() => setGenerating(false))
  }

  const coachingClips = [...(report?.worst_shots || []), ...(report?.best_shots || [])].map((s) => ({
    label: `${s.category} (${s.deviation_score > 0 ? '+' : ''}${s.deviation_score.toFixed(2)})`,
    timestampS: s.timestamp_s, onSeek,
  }))

  const players = position?.players || {}
  const [p1, p2] = Object.keys(players)
  const shotPlayers = shots?.players || {}

  return (
    <div className="rally-analysis-panel">
      <h3>Rally Analysis</h3>

      {p1 && p2 && (
        <div className="panel">
          <h2>Dominance</h2>
          <DominanceBar
            label="Forward Pressure"
            leftLabel={`Player ${p1}`} rightLabel={`Player ${p2}`}
            leftValue={players[p1].forward_pressure_dominance} rightValue={players[p2].forward_pressure_dominance}
          />
          <DominanceBar
            label="Kitchen Arrival"
            leftLabel={`Player ${p1}`} rightLabel={`Player ${p2}`}
            leftValue={players[p1].kitchen_dominance} rightValue={players[p2].kitchen_dominance}
          />
        </div>
      )}

      <div className="panel">
        <h2>Kitchen Arrival %</h2>
        <div className="row" style={{ gap: 20 }}>
          {Object.keys(players).map((pid) => (
            <StatBlock key={pid} value={players[pid].kitchen_arrival_pct} label="Kitchen arrival"
                        playerLabel={`Player ${pid}`} />
          ))}
        </div>
      </div>

      <div className="panel">
        <h2>Shot Distribution</h2>
        <div className="row" style={{ gap: 20 }}>
          {Object.keys(shotPlayers).map((pid) => (
            <React.Fragment key={pid}>
              <StatBlock value={shotPlayers[pid].shot_side_pct?.left}
                          label="Left" playerLabel={`Player ${pid}`} />
              <StatBlock value={shotPlayers[pid].shot_side_pct?.right}
                          label="Right" playerLabel={`Player ${pid}`} />
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="row spread">
          <h2>Coaching Report</h2>
          <button onClick={onGenerateReport} disabled={generating}>
            {generating ? 'Generating…' : report ? 'Regenerate report' : 'Generate report'}
          </button>
        </div>
        {report?.narrative && <div className="coaching-narrative">{report.narrative}</div>}
      </div>

      <ClipList title="Highlights" items={highlightClips} />
      <ClipList title="Coaching Moments" items={coachingClips} />
    </div>
  )
}
