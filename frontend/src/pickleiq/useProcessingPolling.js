import { useEffect, useState } from 'react'

const POLL_INTERVAL_MS = 3000
const DEFAULT_ACTIVE_STATUSES = ['queued', 'running', 'processing']

export default function useProcessingPolling(loadState, initialState,
  activeStatuses = DEFAULT_ACTIVE_STATUSES) {
  const [processing, setProcessing] = useState(initialState)
  const [loadError, setLoadError] = useState('')
  const [pollKey, setPollKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    let timer = null
    let controller = null
    const poll = async () => {
      controller = new AbortController()
      try {
        const next = await loadState(controller.signal)
        if (cancelled) return
        setProcessing(next)
        setLoadError('')
        if (activeStatuses.includes(next.status)) {
          timer = window.setTimeout(poll, POLL_INTERVAL_MS)
        }
      } catch (error) {
        if (!cancelled && error.name !== 'AbortError') {
          setLoadError(error.message || 'Processing status could not be loaded.')
        }
      }
    }
    poll()
    return () => {
      cancelled = true
      controller?.abort()
      if (timer) window.clearTimeout(timer)
    }
  }, [loadState, pollKey, activeStatuses])

  const refresh = ({ reset = false } = {}) => {
    if (reset) setProcessing(initialState)
    setLoadError('')
    setPollKey((key) => key + 1)
  }

  return { processing, loadError, refresh }
}
