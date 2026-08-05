import React, { useEffect, useRef, useState } from 'react'
import { resolveRallySkipTarget } from '../processingAdapter.js'

export default function VideoPreview({ previewUrl, localFile, previewKind, playerRef,
  skipSegments = null }) {
  const [localUrl, setLocalUrl] = useState(null)
  const videoUrl = previewUrl || localUrl
  const internalRef = useRef(null)
  const skipSegmentsRef = useRef(skipSegments)
  skipSegmentsRef.current = skipSegments

  useEffect(() => {
    if (!localFile) {
      setLocalUrl(null)
      return undefined
    }
    const nextUrl = URL.createObjectURL(localFile)
    setLocalUrl(nextUrl)
    return () => URL.revokeObjectURL(nextUrl)
  }, [localFile])

  const setRefs = (node) => {
    internalRef.current = node
    if (playerRef) playerRef.current = node
  }

  const onTimeUpdate = () => {
    const video = internalRef.current
    const segments = skipSegmentsRef.current
    if (!video || !segments?.length) return
    const target = resolveRallySkipTarget(segments, video.currentTime)
    if (target === 'end') {
      video.pause()
    } else if (typeof target === 'number' && target > video.currentTime) {
      video.currentTime = target
    }
  }

  const skipping = Boolean(skipSegments?.length)
  return (
    <figure className="pickleiq-video-preview">
      {videoUrl ? (
        <video ref={setRefs} src={videoUrl} controls preload="metadata"
          onTimeUpdate={onTimeUpdate} aria-label="Video preview" />
      ) : (
        <div className="pickleiq-video-placeholder" role="img" aria-label="Video preview unavailable" />
      )}
      {skipping && videoUrl ? (
        <figcaption>Rally-only playback — dead time between rallies is skipped</figcaption>
      ) : previewKind === 'original-local' && videoUrl ? (
        <figcaption>Original video preview</figcaption>
      ) : null}
    </figure>
  )
}
