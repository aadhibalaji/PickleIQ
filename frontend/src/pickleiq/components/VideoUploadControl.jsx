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
      <label className="pickleiq-upload-control" htmlFor={inputId} aria-disabled={disabled || undefined}>
        <span className="pickleiq-upload-icon" aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v12" /><path d="M7 8l5-5 5 5" /><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
        </span>
        <span className="pickleiq-upload-title" title={file?.name}>
          {file ? file.name : 'Drop your rally video'}
        </span>
        <span className="pickleiq-upload-subtitle">
          {file ? 'Selected — click to choose a different file' : 'or click to choose a file'}
        </span>
      </label>
    </div>
  )
}
