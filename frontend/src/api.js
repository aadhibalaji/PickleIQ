const j = async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText)
  return r.json()
}

export const api = {
  uploadVideo: (file, matchType, onProgress, { signal } = {}) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      const form = new FormData()
      form.append('file', file)
      form.append('match_type', matchType)
      xhr.open('POST', '/api/videos')
      const abort = () => xhr.abort()
      signal?.addEventListener('abort', abort, { once: true })
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && e.total > 0) onProgress?.(e.loaded / e.total)
      }
      xhr.onload = () =>
        xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(xhr.responseText))
      xhr.onerror = () => reject(new Error('upload failed'))
      xhr.onabort = () => reject(new DOMException('Upload aborted', 'AbortError'))
      xhr.onloadend = () => signal?.removeEventListener('abort', abort)
      xhr.send(form)
    }),
  listVideos: () => fetch('/api/videos').then(j),
  deleteVideo: (id) => fetch(`/api/videos/${id}`, { method: 'DELETE' }).then(j),
  getVideo: (id, options) => fetch(`/api/videos/${id}`, options).then(j),
  updateVideo: (id, body, options = {}) =>
    fetch(`/api/videos/${id}`, {
      ...options, method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),
  reprocess: (id, which) => fetch(`/api/videos/${id}/reprocess?which=${which}`, { method: 'POST' }).then(j),
  syncFromCloud: (id) => fetch(`/api/videos/${id}/sync-from-cloud`, { method: 'POST' }).then(j),
  pollCloudWindows: (id) => fetch(`/api/videos/${id}/poll-cloud-windows`, { method: 'POST' }).then(j),
  cancelJob: (jobId) => fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' }).then(j),
  cancelVideoJobs: (id) => fetch(`/api/videos/${id}/cancel`, { method: 'POST' }).then(j),
  streamUrl: (id) => `/api/videos/${id}/stream`,

  getSelfStop: () => fetch('/api/self-stop').then(j),
  setSelfStop: (enabled) =>
    fetch('/api/self-stop', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }),
    }).then(j),

  getRallies: (id, options) => fetch(`/api/videos/${id}/rallies`, options).then(j),
  getRefined: (id, regenerate = 0, options) =>
    fetch(`/api/videos/${id}/refined${regenerate ? '?regenerate=1' : ''}`, options).then(j),
  listWindowGrades: (id) => fetch(`/api/videos/${id}/window_grades`).then(j),
  addWindowGrade: (id, body) =>
    fetch(`/api/videos/${id}/window_grades`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),
  deleteWindowGrade: (gradeId) => fetch(`/api/window_grades/${gradeId}`, { method: 'DELETE' }).then(j),
  getGradeRecommendation: (id) => fetch(`/api/videos/${id}/grade_recommendation`).then(j),
  writeRefineOverride: (body) =>
    fetch('/api/refine/config/override', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),
  putHumanLabels: (id, labels) =>
    fetch(`/api/videos/${id}/labels/human`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(labels),
    }).then(j),
  writeServeDataset: (id) =>
    fetch(`/api/videos/${id}/serve_dataset`, { method: 'POST' }).then(j),

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
  updateBallCorrection: (cid, body) =>
    fetch(`/api/ball_corrections/${cid}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),
  deleteBallCorrection: (cid) => fetch(`/api/ball_corrections/${cid}`, { method: 'DELETE' }).then(j),
  projectPoint: (id, x, y) => fetch(`/api/videos/${id}/project_point?x=${x}&y=${y}`).then(j),

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

  getPass1Candidates: (id) => fetch(`/api/rally-analysis/${id}/pass1-candidates`).then(j),
  listSelections: (id) => fetch(`/api/rally-analysis/${id}/selections`).then(j),
  createSelections: (id, items) =>
    fetch(`/api/rally-analysis/${id}/selections`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(items),
    }).then(j),
  processSelection: (id, selectionId) =>
    fetch(`/api/rally-analysis/${id}/selections/${selectionId}/process`, { method: 'POST' }).then(j),

  getCoachingReport: (id, options) => fetch(`/api/rally-analysis/${id}/coaching-report`, options).then(j),
  getCoachingDebug: (id, focus) =>
    fetch(`/api/rally-analysis/${id}/coaching-debug${focus ? `?focus=${focus}` : ''}`).then(j),
  getContactEvents: (id) => fetch(`/api/rally-analysis/${id}/contact-events`).then(j),
  getReportStats: (id) => fetch(`/api/rally-analysis/${id}/report-stats`).then(j),
  getTrajectories: (id) => fetch(`/api/rally-analysis/${id}/trajectories`).then(j),
  generateCoachingReport: (id, options = {}) =>
    fetch(`/api/rally-analysis/${id}/coaching-report`, { ...options, method: 'POST' }).then(j),
  detectContacts: (id, options = {}) =>
    fetch(`/api/rally-analysis/${id}/detect-contacts`, { ...options, method: 'POST' }).then(j),
  processSingleRally: (id) =>
    fetch(`/api/rally-analysis/${id}/process-single-rally`, { method: 'POST' }).then(j),
  // Expert ground-truth labels on coaching moments (see db.coaching_marks).
  // Distinct from tip-feedback below: that rates tip COPY, these rate the PLAY
  // — and unlike tip feedback they can be recorded for moments no tip covered,
  // which is what makes detector recall measurable.
  listCoachingMarks: (id, focus) =>
    fetch(`/api/rally-analysis/${id}/coaching-marks${focus ? `?focus=${focus}` : ''}`).then(j),
  addCoachingMark: (id, body) =>
    fetch(`/api/rally-analysis/${id}/coaching-marks`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),
  deleteCoachingMark: (markId) =>
    fetch(`/api/rally-analysis/coaching-marks/${markId}`, { method: 'DELETE' }).then(j),

  getTipFeedback: (id) => fetch(`/api/rally-analysis/${id}/tip-feedback`).then(j),
  sendTipFeedback: (id, body) =>
    fetch(`/api/rally-analysis/${id}/tip-feedback`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(j),

  chat: (videoId, messages) =>
    fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, messages }),
    }).then(j),
}
