import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'

const ALL_PLAYERS = 'all'

export default function RallyAnalysisPanel({ videoId, video, onUpdateVideo }) {
  const [report, setReport] = useState(null)
  const [generating, setGenerating] = useState(false)
  const [selectedPlayer, setSelectedPlayer] = useState(ALL_PLAYERS)
  const [selectedTip, setSelectedTip] = useState(null)

  useEffect(() => {
    api.getCoachingReport(videoId).then((r) => setReport(r.report)).catch(() => setReport(null))
    setSelectedPlayer(ALL_PLAYERS)
    setSelectedTip(null)
  }, [videoId])

  const onGenerateReport = () => {
    setGenerating(true)
    api.generateCoachingReport(videoId).then((r) => {
      setReport(r)
      setSelectedPlayer(ALL_PLAYERS)
      setSelectedTip(null)
    }).catch(() => {}).finally(() => setGenerating(false))
  }

  const tips = report?.tips || []
  const players = report?.players || []

  return (
    <div className="rally-analysis-panel">
      <div className="row spread">
        <h3 style={{ margin: 0 }}>Rally Analysis</h3>
        <div className="row">
          <span className="muted">Match type</span>
          <button className={video?.match_type === 'singles' ? 'active' : ''}
                  style={video?.match_type === 'singles' ? { borderColor: '#58a6ff', color: '#58a6ff' } : undefined}
                  onClick={() => onUpdateVideo?.({ match_type: 'singles' })}>
            Singles
          </button>
          <button className={video?.match_type === 'doubles' ? 'active' : ''}
                  style={video?.match_type === 'doubles' ? { borderColor: '#58a6ff', color: '#58a6ff' } : undefined}
                  onClick={() => onUpdateVideo?.({ match_type: 'doubles' })}>
            Doubles
          </button>
        </div>
      </div>

      <div className="panel">
        <div className="row spread">
          <h2>Coaching</h2>
          <button onClick={onGenerateReport} disabled={generating}>
            {generating ? 'Generating…' : report ? 'Regenerate report' : 'Generate report'}
          </button>
        </div>

        {!report ? (
          <div className="clip-list-empty">No coaching report yet.</div>
        ) : (
          <>
            {players.length > 1 && (
              <div className="tabs" style={{ margin: '4px 0 8px' }}>
                <button className={selectedPlayer === ALL_PLAYERS ? 'active' : ''}
                        onClick={() => setSelectedPlayer(ALL_PLAYERS)}>
                  All players
                </button>
                {players.map((p) => (
                  <button key={p} className={selectedPlayer === p ? 'active' : ''}
                          onClick={() => setSelectedPlayer(p)}>
                    {p}
                  </button>
                ))}
              </div>
            )}
            <CoachingVideo videoId={videoId} tips={tips} selectedPlayer={selectedPlayer}
                           selectedTip={selectedTip} onSelectTip={setSelectedTip} />
          </>
        )}
      </div>
    </div>
  )
}

// Self-contained video + pin scrubber, independent of the shared Workspace
// player (separate <video> element, own play/seek state) so the coaching
// view can be scrubbed/pinned without disturbing Setup/Pass tab playback.
function CoachingVideo({ videoId, tips, selectedPlayer, selectedTip, onSelectTip }) {
  const videoRef = useRef(null)
  const [duration, setDuration] = useState(0)

  const visibleTips = selectedPlayer === ALL_PLAYERS
    ? tips
    : tips.filter((t) => t.player_label === selectedPlayer)

  const seekToTip = (t) => {
    const v = videoRef.current
    if (v) v.currentTime = t.timestamp_s
    onSelectTip(t)
  }

  const onScrubClick = (e) => {
    const v = videoRef.current
    if (!v || !duration) return
    const rect = e.currentTarget.getBoundingClientRect()
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    v.currentTime = frac * duration
  }

  return (
    <div className="rally-video">
      <video ref={videoRef} src={api.streamUrl(videoId)} controls preload="metadata"
             onLoadedMetadata={(e) => setDuration(e.target.duration || 0)} />
      <div className="pin-track" onClick={onScrubClick}>
        {visibleTips.map((t) => (
          <button key={t.contact_id} type="button"
                  className={`pin tip-${t.tone}${selectedTip?.contact_id === t.contact_id ? ' pin-active' : ''}`}
                  style={{ left: `${duration ? (t.timestamp_s / duration) * 100 : 0}%` }}
                  title={`${t.player_label} · ${formatCategory(t.category)} · ${formatTime(t.timestamp_s)}`}
                  onClick={(e) => { e.stopPropagation(); seekToTip(t) }} />
        ))}
      </div>

      {selectedTip ? (
        <div className={`tip-panel tip-${selectedTip.tone}`}>
          <div className="row spread">
            <span className="tip-card-badge">
              {selectedTip.tone === 'praise' ? '👍' : '🎯'} {formatCategory(selectedTip.category)}
            </span>
            <span className="tip-panel-player">{selectedTip.player_label}</span>
          </div>
          <p className="tip-card-text">{selectedTip.tip}</p>
        </div>
      ) : (
        <div className="clip-list-empty">
          {visibleTips.length ? "Click a pin to see that shot's feedback." : 'No tips for this player.'}
        </div>
      )}
    </div>
  )
}

function formatCategory(category) {
  return (category || '').replace(/_/g, ' ')
}

function formatTime(s) {
  if (s == null) return ''
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}
