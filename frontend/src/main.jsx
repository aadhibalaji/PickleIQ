import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import PickleIQApp from './pickleiq/PickleIQApp.jsx'
import './styles.css'
import './pickleiq/pickleiq.css'

// PickleIQ (the end-user app) is TEMPORARILY DISABLED as the default while
// coaching feedback quality is under active work — the internal testing
// workspace is the default again. PickleIQ is still reachable explicitly at
// /pickleiq for spot-checking; flip the condition back to re-enable it as
// the default once ready.
const path = window.location.pathname.replace(/\/$/, '')
const isPickleIQRoute = path === '/pickleiq'
if (isPickleIQRoute) document.title = 'PickleIQ'

createRoot(document.getElementById('root')).render(isPickleIQRoute ? <PickleIQApp /> : <App />)
