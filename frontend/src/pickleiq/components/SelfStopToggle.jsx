import React, { useEffect, useState } from 'react'
import { api } from '../../api.js'

// Same EC2 self-stop-on-empty-queue toggle as the internal dev tool
// (Workspace.jsx), surfaced here so it doesn't require switching to Testing
// mode just to flip it.
export default function SelfStopToggle() {
  const [enabled, setEnabled] = useState(null) // null = not loaded yet

  useEffect(() => { api.getSelfStop().then((r) => setEnabled(r.enabled)).catch(() => {}) }, [])

  if (enabled === null) return null

  const toggle = () => {
    const next = !enabled
    setEnabled(next)
    api.setSelfStop(next).catch(() => setEnabled(!next))
  }

  return (
    <label className="pickleiq-self-stop"
      title="EC2 self-stop: automatically stop the worker instance when idle">
      <input type="checkbox" checked={enabled} onChange={toggle} />
      <span>EC2 self-stop</span>
    </label>
  )
}
