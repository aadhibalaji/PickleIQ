import React from 'react'

function formatTimestamp(seconds) {
  const total = Math.max(0, Math.round(seconds))
  const minutes = Math.floor(total / 60)
  const secs = total % 60
  return `${minutes}:${String(secs).padStart(2, '0')}`
}

// Proportional start-to-end timeline of the whole video (like the testing
// workspace's Timeline): each rally is highlighted at its actual position.
// Pass 2 additions: rejected windows render struck-through, and kept windows
// with refined (snapped/split) boundaries draw the refined segments on top of
// a faint full-window backdrop so "altered" rallies show their real timeframe.
export default function RallyTimeline({ sections = [], fps = 30, durationSeconds,
  onSeek, emptyLabel = 'No rallies yet' }) {
  const safeFps = Number(fps) > 0 ? Number(fps) : 30
  const spans = sections
    .filter((s) => Number.isFinite(Number(s.startFrame)) && Number.isFinite(Number(s.endFrame)))
  const maxEnd = spans.reduce(
    (acc, s) => Math.max(acc, Number(s.endFrame) / safeFps), 0)
  const duration = Number(durationSeconds) > 0 ? Number(durationSeconds) : maxEnd || 1

  const pct = (seconds) => `${Math.min(100, Math.max(0, (seconds / duration) * 100))}%`
  const widthPct = (startS, endS) =>
    `calc(${Math.min(100, Math.max(0, ((endS - startS) / duration) * 100))}% )`

  const renderSpan = (key, startS, endS, decision, label, faint = false) => {
    const cls = `pickleiq-timeline-span is-${decision}${faint ? ' is-faint' : ''}`
    const style = { left: pct(startS), width: widthPct(startS, endS) }
    const title = `${label}, ${formatTimestamp(startS)}–${formatTimestamp(endS)}`
    return onSeek && !faint ? (
      <button key={key} type="button" className={cls} style={style} title={title}
        aria-label={`${title}. Seek to section.`} onClick={() => onSeek(startS)} />
    ) : (
      <span key={key} className={cls} style={style} aria-label={title} title={title} />
    )
  }

  return (
    <div className="pickleiq-timeline" aria-live="polite">
      <div className="pickleiq-timeline-track" role="img"
        aria-label={spans.length
          ? `Timeline with ${spans.length} rally section${spans.length === 1 ? '' : 's'}`
          : emptyLabel}>
        {spans.map((section, index) => {
          const startS = Number(section.startFrame) / safeFps
          const endS = Number(section.endFrame) / safeFps
          const decision = section.decision || 'kept'
          const refined = (section.refinedSegments || [])
            .filter((seg) => Number.isFinite(Number(seg.start_frame))
              && Number.isFinite(Number(seg.end_frame)))
          const key = section.id || `${section.startFrame}-${index}`
          const label = decision === 'rejected' ? 'Removed section'
            : decision === 'unknown' ? 'Pending section' : 'Rally'

          if (decision === 'kept' && refined.length) {
            // Altered rally: faint backdrop = original Pass 1 window,
            // solid segments = the refined boundaries that survived.
            return [
              renderSpan(`${key}-backdrop`, startS, endS, 'kept', 'Original detection window', true),
              ...refined.map((seg, segIndex) => renderSpan(
                `${key}-seg-${segIndex}`,
                Number(seg.start_frame) / safeFps,
                Number(seg.end_frame) / safeFps,
                'kept', 'Rally (refined)')),
            ]
          }
          return renderSpan(key, startS, endS, decision, label)
        })}
        {spans.length === 0 && (
          <span className="pickleiq-timeline-empty">{emptyLabel}</span>
        )}
      </div>
      <div className="pickleiq-timeline-scale" aria-hidden="true">
        <span>0:00</span>
        <span>{formatTimestamp(duration)}</span>
      </div>
    </div>
  )
}
