import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import PickleIQApp from './pickleiq/PickleIQApp.jsx'
import './styles.css'
import './pickleiq/pickleiq.css'

// PickleIQ (the end-user app) is the main frontend; the internal testing
// workspace lives at /testing. /pickleiq still works as an alias.
const path = window.location.pathname.replace(/\/$/, '')
const isTestingRoute = path === '/testing'
if (!isTestingRoute) document.title = 'PickleIQ'

createRoot(document.getElementById('root')).render(isTestingRoute ? <App /> : <PickleIQApp />)
