import React from 'react'

// Multi-track rally timeline over the original video's frame axis.
// tracks: [{ name, color, spans: [{start_frame, end_frame}] }]
// marks (optional): ground-truth ball marks [{frame, event_type}] drawn as a
// dot row so coverage is visible at a glance while marking.
const MARK_COLORS = { bounce: '#00e5ff', hit: '#e5a53a', position: '#c3ce4a' }

export default function Timeline({ frameCount, frame, tracks, marks, onSeek }) {
  const W = 1200
  const rowH = 22
  const markRow = marks?.length ? 12 : 0
  const H = tracks.length * rowH + markRow + 26
  const x = (f) => (f / Math.max(frameCount, 1)) * W

  const seek = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const fx = ((e.clientX - rect.left) / rect.width) * frameCount
    onSeek?.(Math.max(0, Math.min(frameCount - 1, Math.round(fx))))
  }

  return (
    <svg className="timeline" viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block', background: '#12171d', borderRadius: 8 }}
         onClick={seek}>
      {tracks.map((t, i) => (
        <g key={t.name}>
          <text x="6" y={i * rowH + 15} fill="#93a4b5" fontSize="11">{t.name}</text>
          <line x1="0" x2={W} y1={(i + 1) * rowH} y2={(i + 1) * rowH} stroke="#1f2831" />
          {t.spans.map((s, j) => (
            <rect key={j} x={x(s.start_frame)} y={i * rowH + 4}
                  width={Math.max(2, x(s.end_frame) - x(s.start_frame))} height={rowH - 8}
                  rx="3" fill={t.color} opacity="0.75">
              <title>{`${t.name}: ${s.start_frame} → ${s.end_frame}`}</title>
            </rect>
          ))}
        </g>
      ))}
      {marks?.length > 0 && marks.map((m, i) => (
        <circle key={m.id || i} cx={x(m.frame)} cy={tracks.length * rowH + 6} r="3"
                fill={MARK_COLORS[m.event_type] || MARK_COLORS.position} opacity="0.9">
          <title>{`${m.event_type || 'position'} @ f${m.frame}`}</title>
        </circle>
      ))}
      <line x1={x(frame)} x2={x(frame)} y1="0" y2={H - 14} stroke="#e8edf2" strokeWidth="1.5" />
      <text x={Math.min(x(frame) + 4, W - 60)} y={H - 2} fill="#e8edf2" fontSize="11">f {frame}</text>
    </svg>
  )
}
