import React, { useState } from 'react'
import App from './App.jsx'
import PickleIQApp from './pickleiq/PickleIQApp.jsx'

// Lets the same session flip between the PickleIQ consumer flow and the
// internal dev/testing tool (App.jsx — Home + Workspace, unchanged) without
// a page reload. /testing still deep-links straight into the dev tool.
const isTestingRoute = window.location.pathname.replace(/\/$/, '') === '/testing'

const barStyle = {
  position: 'fixed', top: 8, right: 8, zIndex: 1000,
  background: '#fff', border: '1px solid #ccc', borderRadius: 6,
  padding: '4px 6px', display: 'flex', gap: 6, boxShadow: '0 1px 4px rgba(0,0,0,0.15)',
}
const btnStyle = (active) => ({
  fontWeight: active ? 'bold' : 'normal', textDecoration: active ? 'underline' : 'none',
})

// The consumer redesign folds its own "Testing" tab into PickleIQApp's
// header (see Header.jsx) instead of the floating corner bar below — this
// is that same button's exact style, exported so both call sites render an
// identical, unstyled-by-the-redesign tab.
export const TESTING_TAB_STYLE = btnStyle(false)

export default function AppRoot() {
  const [mode, setMode] = useState(isTestingRoute ? 'testing' : 'consumer')

  // Both stay mounted at all times — switching modes only toggles which is
  // visible (display:none), so in-progress state (which video, which stage,
  // upload/poll timers) survives flipping back and forth instead of being
  // torn down and reset on every switch.
  return (
    <>
      {/* Only needed as a way back to PickleIQ while inside Testing — the
          consumer flow's own header carries the Testing tab now. */}
      {mode === 'testing' && (
        <div style={barStyle}>
          <button style={btnStyle(false)} onClick={() => setMode('consumer')}>PickleIQ</button>
          <button style={btnStyle(true)} onClick={() => setMode('testing')}>Testing</button>
        </div>
      )}
      <div style={{ display: mode === 'consumer' ? 'block' : 'none' }}>
        <PickleIQApp onSwitchToTesting={() => setMode('testing')} />
      </div>
      <div style={{ display: mode === 'testing' ? 'block' : 'none' }}>
        <App />
      </div>
    </>
  )
}
