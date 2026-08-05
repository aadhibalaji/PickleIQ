import React from 'react'
import { createRoot } from 'react-dom/client'
import AppRoot from './AppRoot.jsx'
import './styles.css'
import './pickleiq/pickleiq.css'

document.title = 'PickleIQ'

createRoot(document.getElementById('root')).render(<AppRoot />)
