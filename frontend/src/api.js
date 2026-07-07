const j = async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText)
  return r.json()
}

export const api = {
  uploadVideo: (file, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      const form = new FormData()
      form.append('file', file)
      xhr.open('POST', '/api/videos')
      xhr.upload.onprogress = (e) => onProgress?.(e.loaded / e.total)
      xhr.onload = () =>
        xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(xhr.responseText))
      xhr.onerror = () => reject(new Error('upload failed'))
      xhr.send(form)
    }),
  listVideos: () => fetch('/api/videos').then(j),
  getVideo: (id) => fetch(`/api/videos/${id}`).then(j),
  reprocess: (id, which) => fetch(`/api/videos/${id}/reprocess?which=${which}`, { method: 'POST' }).then(j),
  cancelJob: (jobId) => fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' }).then(j),
  cancelVideoJobs: (id) => fetch(`/api/videos/${id}/cancel`, { method: 'POST' }).then(j),
  streamUrl: (id) => `/api/videos/${id}/stream`,

  getRallies: (id) => fetch(`/api/videos/${id}/rallies`).then(j),
  putHumanLabels: (id, labels) =>
    fetch(`/api/videos/${id}/labels/human`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(labels),
    }).then(j),

  getOverlays: (id, source, start, end) =>
    fetch(`/api/videos/${id}/overlays?source=${source}&start=${start}&end=${end}`).then(j),

  referencePoints: () => fetch('/api/calibration/reference_points').then(j),
  getCalibration: (videoId) =>
    fetch(`/api/calibration${videoId ? `?video_id=${videoId}` : ''}`).then(j),
  saveCalibration: (body) =>
    fetch('/api/calibration', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),
  suggestCorners: (id, frame) => fetch(`/api/videos/${id}/suggest_corners?frame=${frame}`).then(j),
  projectCourt: (calId) => fetch(`/api/calibration/${calId}/project`).then(j),

  listNotes: (id) => fetch(`/api/videos/${id}/notes`).then(j),
  addNote: (id, body) =>
    fetch(`/api/videos/${id}/notes`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),
  deleteNote: (noteId) => fetch(`/api/notes/${noteId}`, { method: 'DELETE' }).then(j),

  listBallCorrections: (id) => fetch(`/api/videos/${id}/ball_corrections`).then(j),
  addBallCorrection: (id, body) =>
    fetch(`/api/videos/${id}/ball_corrections`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),

  generateReport: (id, aiSource) =>
    fetch(`/api/videos/${id}/report`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ai_source: aiSource }),
    }).then(j),
  listReports: (id) => fetch(`/api/videos/${id}/reports`).then(j),
  applySuggestions: (reportId, suggestions) =>
    fetch(`/api/reports/${reportId}/apply_suggestions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ suggestions }),
    }).then(j),
  getParams: () => fetch('/api/params').then(j),
  getScoreboard: () => fetch('/api/learning/scoreboard').then(j),
  getPendingChanges: () => fetch('/api/pending_changes').then(j),
  approvePendingChange: (id) => fetch(`/api/pending_changes/${id}/approve`, { method: 'POST' }).then(j),
  rejectPendingChange: (id) => fetch(`/api/pending_changes/${id}/reject`, { method: 'POST' }).then(j),
  detectDebug: (id) => fetch(`/api/videos/${id}/detect_debug`).then(j),
  getModels: (kind) => fetch(`/api/models${kind ? `?kind=${kind}` : ''}`).then(j),
  promoteModel: (version) => fetch(`/api/models/${version}/promote`, { method: 'POST' }).then(j),
  rejectModel: (version) => fetch(`/api/models/${version}/reject`, { method: 'POST' }).then(j),
  getLearningCurve: (kind) => fetch(`/api/models/learning_curve${kind ? `?kind=${kind}` : ''}`).then(j),

  chat: (videoId, messages) =>
    fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, messages }),
    }).then(j),
}
