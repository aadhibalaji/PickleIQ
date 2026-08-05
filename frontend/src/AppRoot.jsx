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

export default function AppRoot() {
  const [mode, setMode] = useState(isTestingRoute ? 'testing' : 'consumer')

  return (
    <>
      <div style={barStyle}>
        <button style={btnStyle(mode === 'consumer')} onClick={() => setMode('consumer')}>PickleIQ</button>
        <button style={btnStyle(mode === 'testing')} onClick={() => setMode('testing')}>Testing</button>
      </div>
      {mode === 'consumer' ? <PickleIQApp /> : <App />}
    </>
  )
}
