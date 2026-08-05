import React, { useId } from 'react'

const ACCEPTED_VIDEO_FORMATS = 'video/mp4,video/quicktime,video/webm,video/x-msvideo,video/x-matroska'

export default function VideoUploadControl({ file, onChange, onError, disabled = false }) {
  const inputId = useId()
  return (
    <div className="pickleiq-upload-field">
      <input id={inputId} className="pickleiq-visually-hidden" type="file"
        accept={ACCEPTED_VIDEO_FORMATS} disabled={disabled}
        onChange={(event) => {
          const selected = event.target.files?.[0] ?? null
          const supportedExtension = /\.(mp4|mov|webm|avi|mkv)$/i.test(selected?.name || '')
          const supported = !selected || selected.type.startsWith('video/') || supportedExtension
          if (supported) onChange(selected)
          else onError?.('Choose a supported video file (MP4, MOV, WebM, AVI, or MKV).')
          event.target.value = ''
        }} />
      <label className="pickleiq-upload-control" htmlFor={inputId} aria-label="Choose a video file"
        aria-disabled={disabled || undefined}>
        <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
          <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4.5A1.5 1.5 0 006.5 20h11a1.5 1.5 0 001.5-1.5V14" />
        </svg>
      </label>
      <p className={`pickleiq-file-name${file ? ' has-file' : ''}`} aria-live="polite" title={file?.name}>
        {file?.name || 'Choose video'}
      </p>
    </div>
  )
}
