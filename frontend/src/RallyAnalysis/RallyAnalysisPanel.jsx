import React from 'react'
import ClipList from './ClipList.jsx'
import DominanceBar from './DominanceBar.jsx'
import StatBlock from './StatBlock.jsx'
import { mockPlayers, mockRallyMetrics, averagePlayerMetric } from './mockData.js'

// TODO(integration pass): replace mockRallyMetrics/mockPlayers with
// api.getRallyMetrics(videoId) / api.getCoachingReport(videoId).
export default function RallyAnalysisPanel({ videoId, onSeek }) {
  const [p1, p2] = Object.keys(mockPlayers)
  const kitchenArrival = averagePlayerMetric(mockRallyMetrics, 'kitchen_arrival')
  const forwardPressure = averagePlayerMetric(mockRallyMetrics, 'forward_pressure')

  return (
    <div className="rally-analysis-panel">
      <h3>Rally Analysis</h3>

      <div className="panel">
        <h2>Dominance</h2>
        <DominanceBar
          label="Forward Pressure"
          leftLabel={mockPlayers[p1].label} rightLabel={mockPlayers[p2].label}
          leftValue={forwardPressure[p1]} rightValue={forwardPressure[p2]}
        />
        <DominanceBar
          label="Kitchen Arrival"
          leftLabel={mockPlayers[p1].label} rightLabel={mockPlayers[p2].label}
          leftValue={kitchenArrival[p1]} rightValue={kitchenArrival[p2]}
        />
      </div>

      <div className="panel">
        <h2>Kitchen Arrival %</h2>
        <div className="row" style={{ gap: 20 }}>
          {Object.keys(mockPlayers).map((pid) => (
            <StatBlock key={pid} value={kitchenArrival[pid]} label="Kitchen arrival"
                        playerLabel={mockPlayers[pid].label} />
          ))}
        </div>
      </div>

      {/* Shot-distribution (left/right %) — real data lands in a later wave;
          stubbed here as a StatBlock per side so the layout is ready. */}
      <div className="panel">
        <h2>Shot Distribution (stub)</h2>
        <div className="muted" style={{ marginBottom: 8 }}>
          Left/right shot split per player — populated once shot_side_pct is wired up.
        </div>
        <div className="row" style={{ gap: 20 }}>
          {Object.keys(mockPlayers).map((pid) => (
            <React.Fragment key={pid}>
              <StatBlock value={mockRallyMetrics[0].shot_side_pct[pid]?.left}
                          label="Left" playerLabel={mockPlayers[pid].label} />
              <StatBlock value={mockRallyMetrics[0].shot_side_pct[pid]?.right}
                          label="Right" playerLabel={mockPlayers[pid].label} />
            </React.Fragment>
          ))}
        </div>
      </div>

      <ClipList title="Highlights" items={[]} />
      <ClipList title="Coaching Moments" items={[]} />
    </div>
  )
}
