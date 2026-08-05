import React from 'react'

// Per-clip coaching preferences: singles/doubles + feedback focus
// (positioning vs form). Both persist via PATCH /api/videos/{id}
// (onUpdateVideo prop — partial update, one field at a time). Shared between
// the Rally Analysis panel header and the upload flow so the two stay in
// sync visually and behaviorally. Focus is exclusive by design (product
// decision): a report evaluates one metric family, never both.
const ACTIVE_STYLE = { borderColor: '#58a6ff', color: '#58a6ff' }

function Segmented({ label, value, options, onSelect }) {
  return (
    <div className="row">
      <span className="muted">{label}</span>
      {options.map(([key, text]) => (
        <button key={key} className={value === key ? 'active' : ''}
                style={value === key ? ACTIVE_STYLE : undefined}
                onClick={() => onSelect(key)}>
          {text}
        </button>
      ))}
    </div>
  )
}

export default function CoachingSetupSelector({ video, onUpdateVideo }) {
  return (
    <div className="row" style={{ gap: 16 }}>
      <Segmented label="Match type" value={video?.match_type}
                 options={[['singles', 'Singles'], ['doubles', 'Doubles']]}
                 onSelect={(v) => onUpdateVideo?.({ match_type: v })} />
      <Segmented label="Feedback" value={video?.coaching_focus || 'form'}
                 options={[['form', 'Form'], ['positioning', 'Positioning']]}
                 onSelect={(v) => onUpdateVideo?.({ coaching_focus: v })} />
    </div>
  )
}
