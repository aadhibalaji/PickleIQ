import { api } from '../api.js'

export async function startPickleIQAnalysis({ file, gameType, analysisType, onProgress, signal }) {
  const video = await api.uploadVideo(file, gameType, onProgress, { signal })
  await api.updateVideo(video.id, {
    coaching_focus: analysisType,
  }, { signal })
  return video
}

const ACTIVE_STATUSES = new Set(['queued', 'running'])

function latestJob(jobs, kind) {
  return (jobs || [])
    .filter((job) => job.kind === kind)
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))[0] || null
}

function normalizeProgress(job) {
  if (job?.status === 'done') return 1
  if (!ACTIVE_STATUSES.has(job?.status)) return null
  const value = Number(job.progress)
  return Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : null
}

function normalizeJobStatus(job) {
  const status = job?.status || 'queued'
  return ['queued', 'running', 'done', 'error', 'cancelled'].includes(status) ? status : 'unknown'
}

function safeArray(value) { return Array.isArray(value) ? value : [] }

export function normalizePass1ProcessingState(video, rallies) {
  const job = latestJob(video.jobs, 'pass1')
  const status = normalizeJobStatus(job)
  const retainedRallies = safeArray(rallies?.pass1)
    .filter((rally) => rally && Number.isFinite(Number(rally.start_frame))
      && Number.isFinite(Number(rally.end_frame)))
    .map((rally) => ({
      ...rally,
      id: rally.id || `${rally.start_frame}:${rally.end_frame}`,
    }))

  const fps = Number(video?.fps) > 0 ? Number(video.fps) : 30
  return {
    jobId: job?.id || null,
    status,
    progress: normalizeProgress(job),
    message: job?.message || '',
    previewUrl: null,
    previewKind: 'original-local',
    retainedRallies,
    fps,
    durationSeconds: Number(video?.frame_count) > 0 ? Number(video.frame_count) / fps : null,
    error: status === 'error' || status === 'cancelled'
      ? job?.message || `Pass 1 ${status}.`
      : status === 'unknown' ? 'Pass 1 returned an unknown processing status.' : null,
  }
}

export async function getPass1ProcessingState(videoId, signal) {
  let video = await api.getVideo(videoId, { signal })

  if (video.s3_key && video.run_processing === false) {
    await api.pollCloudWindows(videoId)
    await api.syncFromCloud(videoId)
    video = await api.getVideo(videoId, { signal })
  }

  const rallies = await api.getRallies(videoId, { signal })
  return normalizePass1ProcessingState(video, rallies)
}

export async function retryPass1Processing(videoId) {
  return api.reprocess(videoId, 'pass1')
}

// Pass 2 is never auto-chained after Pass 1 (see pipeline.start_processing) —
// someone has to review Pass 1's candidates and explicitly queue Pass 2 for
// the ones to keep (the internal dev tool's Coaching Selections step does
// this; PickleIQ's RallyReviewScreen is the consumer-facing equivalent).
// keptRallyIds are rally_labels ids the user chose to keep; drawnRanges are
// user-marked {start_frame, end_frame} spans Pass 1 missed. Safe to call more
// than once for the same video (e.g. a retry after a network error): skips
// creating selections again if any already exist, and treats "no pending
// selections to process" as success rather than an error.
export async function submitRallySelections(videoId, { keptRallyIds = [], drawnRanges = [] } = {}) {
  const items = [
    ...safeArray(keptRallyIds).filter(Boolean).map((id) => ({ source: 'pass1_accepted', rally_label_id: id })),
    ...safeArray(drawnRanges).map((r) => ({ source: 'user_drawn', start_frame: r.start_frame, end_frame: r.end_frame })),
  ]
  if (!items.length) throw new Error('Select at least one rally before continuing.')
  const existing = await api.listSelections(videoId).catch(() => ({ selections: [] }))
  if (!(existing.selections || []).length) {
    await api.createSelections(videoId, items)
  }
  try {
    await api.processSelections(videoId)
  } catch (error) {
    if (!/no pending selections/i.test(error.message || '')) throw error
  }
}

function normalizeDecision(verdict) {
  if (verdict === 'rally') return 'kept'
  if (verdict === 'rejected') return 'rejected'
  return 'unknown'
}

function normalizePass2Sections(pass1Rallies, windows) {
  const classified = new Map(safeArray(windows).map((window) => [
    `${window.pass1_start_frame}:${window.pass1_end_frame}`,
    window,
  ]))

  const candidates = safeArray(pass1Rallies).map((rally) => {
    const window = classified.get(`${rally.start_frame}:${rally.end_frame}`)
    return {
      id: rally.id || `${rally.start_frame}:${rally.end_frame}`,
      startFrame: rally.start_frame,
      endFrame: rally.end_frame,
      decision: normalizeDecision(window?.ai_verdict),
      rejectionReason: window?.rejection_reason || null,
      refinedSegments: window?.segments || [],
    }
  })

  for (const window of safeArray(windows)) {
    const key = `${window.pass1_start_frame}:${window.pass1_end_frame}`
    if (!candidates.some((section) => `${section.startFrame}:${section.endFrame}` === key)) {
      candidates.push({
        id: key,
        startFrame: window.pass1_start_frame,
        endFrame: window.pass1_end_frame,
        decision: normalizeDecision(window.ai_verdict),
        rejectionReason: window.rejection_reason || null,
        refinedSegments: window.segments || [],
      })
    }
  }

  return candidates.filter((item) => Number.isFinite(Number(item.startFrame))
    && Number.isFinite(Number(item.endFrame)))
    .sort((a, b) => Number(a.startFrame) - Number(b.startFrame))
}

export function normalizePass2ProcessingState(video, rallies, refined, pass1Result) {
  const job = latestJob(video.jobs, 'pass2')
  const status = normalizeJobStatus(job)
  const sections = normalizePass2Sections(
    pass1Result?.retainedRallies || safeArray(rallies?.pass1),
    safeArray(refined?.windows),
  )
  const keptSections = sections.filter((section) => section.decision === 'kept')
  const rejectedSections = sections.filter((section) => section.decision === 'rejected')
  const unknownSections = sections.filter((section) => section.decision === 'unknown')
  const partial = Boolean(refined.partial)
  const incompleteError = status === 'done' && !partial && unknownSections.length > 0
    ? 'Pass 2 finished, but some rally classifications were unavailable.'
    : null

  return {
    jobId: job?.id || null,
    status,
    progress: normalizeProgress(job),
    message: job?.message || '',
    previewUrl: pass1Result?.previewUrl || null,
    previewKind: pass1Result?.previewUrl ? 'pass1-output' : 'original-local',
    sections,
    keptSections,
    rejectedSections,
    unknownSections,
    classificationComplete: !partial && unknownSections.length === 0,
    finalVideoUrl: null,
    fps: video.fps || pass1Result?.fps || 30,
    durationSeconds: Number(video?.frame_count) > 0
      ? Number(video.frame_count) / (Number(video.fps) > 0 ? Number(video.fps) : 30)
      : pass1Result?.durationSeconds || null,
    partial,
    error: (typeof refined?.error === 'string' ? refined.error : null) || (status === 'error' || status === 'cancelled'
      ? job?.message || `Pass 2 ${status}.`
      : status === 'unknown' ? 'Pass 2 returned an unknown processing status.' : incompleteError),
  }
}

export async function getPass2ProcessingState(videoId, pass1Result, signal) {
  let video = await api.getVideo(videoId, { signal })

  if (video.s3_key && video.run_processing === false) {
    await api.pollCloudWindows(videoId)
    await api.syncFromCloud(videoId)
    video = await api.getVideo(videoId, { signal })
  }

  const [rallies, refined] = await Promise.all([
    api.getRallies(videoId, { signal }),
    api.getRefined(videoId, 0, { signal }),
  ])
  return normalizePass2ProcessingState(video, rallies, refined, pass1Result)
}

export async function retryPass2Processing(videoId) {
  return api.reprocess(videoId, 'pass2')
}

function normalizeAnalysisStatus(status) {
  if (status === 'queued') return 'queued'
  if (status === 'running') return 'processing'
  if (status === 'done') return 'ready'
  if (status === 'error' || status === 'cancelled') return 'failed'
  return 'failed'
}

export function normalizeAnalysisPreparationState(video) {
  const job = latestJob(video.jobs, 'contacts')
  const status = normalizeAnalysisStatus(job?.status)
  return {
    jobId: job?.id || null,
    status,
    progress: normalizeProgress(job),
    message: job?.message || '',
    estimatedSecondsRemaining: null,
    error: status === 'failed' ? job?.message || 'Player preparation failed.' : null,
  }
}

export async function getAnalysisPreparationState(videoId, signal) {
  let video = await api.getVideo(videoId, { signal })
  if (video.s3_key && video.run_processing === false) {
    await api.pollCloudWindows(videoId)
    await api.syncFromCloud(videoId)
    video = await api.getVideo(videoId, { signal })
  }
  return normalizeAnalysisPreparationState(video)
}

function timestampLabel(seconds) {
  const safeSeconds = Math.max(0, Math.round(Number(seconds) || 0))
  const minutes = Math.floor(safeSeconds / 60)
  const remainder = safeSeconds % 60
  return `${minutes}:${String(remainder).padStart(2, '0')}`
}

export function normalizeAnalysisResult(report, finalVideoUrl = null) {
  const safeReport = report && typeof report === 'object' ? report : {}
  const generalEntries = safeArray(safeReport.general_feedback).length
    ? safeArray(safeReport.general_feedback)
    : safeArray(safeReport.player_themes).map((entry) => ({ ...entry, theme: entry?.theme || '' }))
  const generalFeedback = generalEntries.filter((entry) => entry && typeof entry === 'object').map((entry, index) => ({
    id: `general-${entry.player_id ?? index}`,
    playerId: entry.player_id ?? null,
    playerLabel: entry.player_label || `Player ${index + 1}`,
    praise: entry.praise || '',
    improve: entry.improve || '',
    theme: entry.theme || '',
  }))
  const tipIdCounts = new Map()
  const timestampFeedback = safeArray(safeReport.tips)
    .filter((tip) => tip && typeof tip === 'object' && tip.timestamp_s !== null
      && tip.timestamp_s !== '' && Number.isFinite(Number(tip.timestamp_s)))
    .map((tip, index) => {
      const sourceTimestampSeconds = Math.max(0, Number(tip.timestamp_s))
      const displayLabel = timestampLabel(sourceTimestampSeconds)
      const baseId = String(tip.contact_id || `tip-${index}`)
      const occurrence = tipIdCounts.get(baseId) || 0
      tipIdCounts.set(baseId, occurrence + 1)
      return {
        id: occurrence ? `${baseId}-${occurrence + 1}` : baseId,
        timestampSeconds: sourceTimestampSeconds,
        timestampLabel: displayLabel,
        displayLabel,
        displayTimestampSeconds: null,
        sourceTimestampSeconds,
        timeline: 'source',
        feedback: tip.tip || '',
        category: tip.category || null,
        tone: tip.tone || null,
        playerId: tip.player_id ?? null,
        playerLabel: tip.player_label || null,
      }
    })
    .sort((a, b) => a.sourceTimestampSeconds - b.sourceTimestampSeconds)
  return {
    status: 'completed',
    progress: 1,
    estimatedSecondsRemaining: null,
    finalVideoUrl,
    finalVideoTimeline: null,
    generalFeedback,
    timestampFeedback,
    resultId: safeReport.video_id || null,
    focus: safeReport.focus || null,
    players: safeArray(safeReport.players),
    rawReport: safeReport,
    error: null,
  }
}

export async function generatePlayerAnalysis(videoId, finalVideoUrl, signal) {
  try {
    const report = await api.generateCoachingReport(videoId, { signal })
    return normalizeAnalysisResult(report, finalVideoUrl)
  } catch (generationError) {
    if (generationError.name === 'AbortError') throw generationError
    const cached = await api.getCoachingReport(videoId, { signal }).catch(() => null)
    if (cached?.report) return normalizeAnalysisResult(cached.report, finalVideoUrl)
    throw generationError
  }
}

export async function retryAnalysisPreparation(videoId, signal) {
  return api.detectContacts(videoId, { signal })
}

// ------------------------------------------------------------------
// Rally-only playback: the results/analysis player shows the ORIGINAL
// video but auto-skips the dead time between kept rallies, so the user
// experiences a "cropped" video without any server-side re-encode and
// every tip timestamp stays on the source timeline.
// Each segment is padded slightly so contacts right on a rally boundary
// (and the moment of the serve) aren't skipped over.
const SEGMENT_PAD_S = 0.75

export function buildRallySegments(sections, fps) {
  const safeFps = Number(fps) > 0 ? Number(fps) : 30
  const spans = safeArray(sections)
    .filter((section) => Number.isFinite(Number(section?.startFrame))
      && Number.isFinite(Number(section?.endFrame)))
    .map((section) => ({
      startSeconds: Math.max(0, Number(section.startFrame) / safeFps - SEGMENT_PAD_S),
      endSeconds: Number(section.endFrame) / safeFps + SEGMENT_PAD_S,
    }))
    .filter((span) => span.endSeconds > span.startSeconds)
    .sort((a, b) => a.startSeconds - b.startSeconds)
  const merged = []
  for (const span of spans) {
    const last = merged[merged.length - 1]
    if (last && span.startSeconds <= last.endSeconds) {
      last.endSeconds = Math.max(last.endSeconds, span.endSeconds)
    } else {
      merged.push({ ...span })
    }
  }
  return merged
}

// Returns where playback should jump given the current position:
// null = inside a rally (or no segments — leave playback alone),
// a number = seek to that second, 'end' = past the last rally (stop).
export function resolveRallySkipTarget(segments, currentSeconds) {
  if (!safeArray(segments).length || !Number.isFinite(currentSeconds)) return null
  for (const segment of segments) {
    if (currentSeconds < segment.startSeconds) return segment.startSeconds
    if (currentSeconds <= segment.endSeconds) return null
  }
  return 'end'
}

// ------------------------------------------------------------------
// Analysis time estimate: neither the contacts job nor the coaching LLM
// call reports fine-grained progress, so the countdown is a heuristic —
// a base cost for contact detection plus a per-rally cost for the
// keyframe + LLM work, clamped to a sane range, counting down with a
// floor so it never claims "0 seconds" while still running.
const ANALYSIS_BASE_S = 60
const ANALYSIS_PER_RALLY_S = 15
const ANALYSIS_MIN_TOTAL_S = 90
const ANALYSIS_MAX_TOTAL_S = 300
const ANALYSIS_FLOOR_S = 10

export function estimateAnalysisTotalSeconds(keptRallyCount) {
  const count = Number.isFinite(Number(keptRallyCount)) ? Math.max(0, Number(keptRallyCount)) : 0
  const total = ANALYSIS_BASE_S + ANALYSIS_PER_RALLY_S * count
  return Math.min(Math.max(total, ANALYSIS_MIN_TOTAL_S), ANALYSIS_MAX_TOTAL_S)
}

export function remainingAnalysisSeconds(totalSeconds, elapsedSeconds) {
  if (!Number.isFinite(totalSeconds) || !Number.isFinite(elapsedSeconds)) return null
  return Math.max(totalSeconds - elapsedSeconds, ANALYSIS_FLOOR_S)
}

export function resolveFeedbackSeekSeconds(entry, displayedTimeline) {
  if (!entry || !displayedTimeline || entry.timeline !== displayedTimeline) return null
  const seconds = displayedTimeline === 'trimmed'
    ? entry.displayTimestampSeconds
    : entry.sourceTimestampSeconds
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null
}
