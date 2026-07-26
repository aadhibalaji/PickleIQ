// Single source of truth for Player 1 / Player 2 colors, reused by the
// heatmap, tips, filter tabs, and report stats so a given player is always
// the same color everywhere in the Rally Analysis panel.
//
// Rule: sort the distinct player_ids seen in the data ascending, assign
// --player1-color to the first and --player2-color to the rest. Rally
// Analysis only ever surfaces two player "slots" per video (singles/doubles
// are pre-merged to 2 by the backend — see coaching.merge_singles_players),
// so this stays a simple 2-token map rather than a full palette.
export function buildPlayerColorMap(playerIds) {
  const sorted = [...new Set(playerIds.filter((id) => id != null))].sort((a, b) => a - b)
  const map = {}
  sorted.forEach((pid, i) => { map[pid] = i === 0 ? 'var(--player1-color)' : 'var(--player2-color)' })
  return map
}

export function playerColor(colorMap, playerId) {
  return colorMap[playerId] || 'var(--muted)'
}

// Tips key players by an int id but display "Player N" labels; pull the id
// back out for callers that only have the label (e.g. the filter tabs).
export function labelToPlayerId(label) {
  const m = /(\d+)/.exec(label || '')
  return m ? Number(m[1]) : null
}
