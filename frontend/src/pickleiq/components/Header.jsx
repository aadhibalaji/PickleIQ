import React from 'react'
import { TESTING_TAB_STYLE } from '../../AppRoot.jsx'
import GlassFilterDefs from './GlassFilterDefs.jsx'

// Persistent nav shown on every PickleIQ consumer screen (mirrors the design
// reference, where this header sits above all four mocked screens
// unchanged). `active` highlights the matching pill; mid-flow screens
// (Pass 1/Review/Pass 2/Analysis) pass no active value since neither pill
// represents "where you are" during a run.
// `onTesting` is the escape hatch back to the internal Testing tool — see
// AppRoot.jsx. Its button reuses AppRoot's exact inline style so the tab
// looks and behaves identically to its old floating-corner incarnation,
// just relocated into the header per the redesign handoff.
export default function Header({ active, onHome, onPastClips, onTesting }) {
  return (
    <header className="pickleiq-header">
      <GlassFilterDefs />
      <button type="button" className="pickleiq-brand" onClick={onHome}>
        <span className="pickleiq-brand-pickle">PICKLE</span><span className="pickleiq-brand-iq">IQ</span>
      </button>
      <div className="pickleiq-header-right">
        <nav className="pickleiq-nav-pills" aria-label="PickleIQ sections">
          <button type="button" className={active === 'home' ? 'is-active' : ''} onClick={onHome}>
            Home
          </button>
          <button type="button" className={active === 'pastclips' ? 'is-active' : ''} onClick={onPastClips}>
            Past Clips
          </button>
        </nav>
        {onTesting && (
          <button type="button" style={TESTING_TAB_STYLE} onClick={onTesting}>Testing</button>
        )}
      </div>
    </header>
  )
}
