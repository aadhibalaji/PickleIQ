// Hardcoded mock data matching the RallyMetrics / CoachingReport shapes in
// backend/app/rallyanalysis/contracts.py (field names mirrored exactly) so
// swapping these for real api.* calls later is a near-zero diff.
// player_id keys are the same PlayerObs.slot values used elsewhere ('1'/'2').

export const mockPlayers = {
  '1': { label: 'Player A', team: 'near' },
  '2': { label: 'Player B', team: 'far' },
}

// RallyMetrics[] — one per rally
export const mockRallyMetrics = [
  {
    rally_id: 'r1', video_id: 'v1',
    kitchen_arrival: { '1': true, '2': false },
    forward_pressure: { '1': 0.72, '2': 0.31 },
    shot_side_pct: { '1': { left: 0.4, right: 0.6 }, '2': { left: 0.55, right: 0.45 } },
    shot_count: 14, rally_length_bucket: 'medium', active_duration_s: 9.4,
  },
  {
    rally_id: 'r2', video_id: 'v1',
    kitchen_arrival: { '1': true, '2': true },
    forward_pressure: { '1': 0.58, '2': 0.49 },
    shot_side_pct: { '1': { left: 0.5, right: 0.5 }, '2': { left: 0.3, right: 0.7 } },
    shot_count: 22, rally_length_bucket: 'long', active_duration_s: 15.1,
  },
  {
    rally_id: 'r3', video_id: 'v1',
    kitchen_arrival: { '1': false, '2': true },
    forward_pressure: { '1': 0.22, '2': 0.81 },
    shot_side_pct: { '1': { left: 0.65, right: 0.35 }, '2': { left: 0.45, right: 0.55 } },
    shot_count: 6, rally_length_bucket: 'short', active_duration_s: 3.8,
  },
]

// CoachingReport — one per video
export const mockCoachingReport = {
  video_id: 'v1',
  category_ratios: {
    footwork: { flagged_count: 5, active_rally_seconds: 28.3, ratio: 0.18 },
    paddle_prep: { flagged_count: 2, active_rally_seconds: 28.3, ratio: 0.07 },
    contact_point: { flagged_count: 3, active_rally_seconds: 28.3, ratio: 0.11 },
    follow_through: { flagged_count: 1, active_rally_seconds: 28.3, ratio: 0.04 },
  },
  worst_shots: [],
  best_shots: [],
  narrative: '',
}

// Average a per-player metric across all rallies (booleans -> arrival rate).
export function averagePlayerMetric(rallies, field) {
  const sums = {}, counts = {}
  for (const r of rallies) {
    for (const [pid, v] of Object.entries(r[field] || {})) {
      const n = typeof v === 'boolean' ? (v ? 1 : 0) : v
      sums[pid] = (sums[pid] || 0) + n
      counts[pid] = (counts[pid] || 0) + 1
    }
  }
  const out = {}
  for (const pid of Object.keys(sums)) out[pid] = sums[pid] / counts[pid]
  return out
}
