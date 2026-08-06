import assert from 'node:assert/strict'
import {
  buildRallySegments,
  estimateAnalysisTotalSeconds,
  normalizeAnalysisPreparationState,
  normalizeAnalysisResult,
  normalizePass1ProcessingState,
  normalizePass2ProcessingState,
  remainingAnalysisSeconds,
  resolveFeedbackSeekSeconds,
  resolveRallySkipTarget,
} from './src/pickleiq/processingAdapter.js'

const video = { fps: 30, jobs: [{ id: 'p1', kind: 'pass1', status: 'done', progress: 4 }] }
const pass1 = normalizePass1ProcessingState(video, { pass1: [] })
assert.equal(pass1.status, 'done')
assert.deepEqual(pass1.retainedRallies, [])
assert.equal(pass1.progress, 1)

const retained = normalizePass1ProcessingState(video, { pass1: [
  { start_frame: 10, end_frame: 40 },
  { start_frame: 'bad', end_frame: 80 },
] })
assert.deepEqual(retained.retainedRallies.map((item) => item.id), ['10:40'])

const unknown = normalizePass1ProcessingState({ jobs: [{ kind: 'pass1', status: 'mystery' }] }, {})
assert.equal(unknown.status, 'unknown')
assert.match(unknown.error, /unknown processing status/)

const pass2 = normalizePass2ProcessingState(
  { fps: 0, jobs: [{ kind: 'pass2', status: 'done' }] },
  { pass1: [] },
  { windows: [
    { pass1_start_frame: 90, pass1_end_frame: 120, ai_verdict: 'rejected' },
    { pass1_start_frame: 30, pass1_end_frame: 60, ai_verdict: 'rally' },
    { pass1_start_frame: 150, pass1_end_frame: 180, ai_verdict: 'surprise' },
  ] },
  null,
)
assert.deepEqual(pass2.sections.map((item) => item.decision), ['kept', 'rejected', 'unknown'])
assert.equal(pass2.classificationComplete, false)
assert.match(pass2.error, /classifications were unavailable/)

const report = normalizeAnalysisResult({
  general_feedback: 'unexpected',
  tips: [
    { contact_id: 'late', timestamp_s: 307, tip: 'Late' },
    { contact_id: 'invalid', timestamp_s: 'not-a-time', tip: 'Invalid' },
    { contact_id: 'early', timestamp_s: 12.4, tip: 'Early' },
    { contact_id: 'early', timestamp_s: 18, tip: 'Second early tip' },
    { contact_id: 'missing', timestamp_s: null, tip: 'Missing timestamp' },
  ],
  players: null,
})
assert.deepEqual(report.timestampFeedback.map((item) => item.id), ['early', 'early-2', 'late'])
assert.deepEqual(report.generalFeedback, [])
assert.equal(resolveFeedbackSeekSeconds(report.timestampFeedback[0], 'source'), 12.4)
assert.equal(resolveFeedbackSeekSeconds(report.timestampFeedback[0], 'trimmed'), null)

// Key Takeaways (summary + takeaways) is the current shape; a legacy
// {praise, improve} report (pre-dating this feature) degrades to a 2-item
// takeaways list instead of showing nothing. Player photos attach by
// player_label to both general feedback entries and matching timestamp tips.
const withTakeaways = normalizeAnalysisResult({
  general_feedback: [
    { player_id: 1, player_label: 'Player 1', summary: 'Strong session overall.',
      takeaways: ['Great footwork', 'Watch paddle prep', ''] },
    { player_id: 2, player_label: 'Player 2', praise: 'Solid net play', improve: 'Move your feet' },
  ],
  player_photos: { 'Player 1': 'AAAA', 'Player 2': 'BBBB' },
  tips: [{ contact_id: 't1', timestamp_s: 5, tip: 'Nice recovery', player_label: 'Player 1' }],
})
assert.deepEqual(withTakeaways.generalFeedback[0].takeaways, ['Great footwork', 'Watch paddle prep'])
assert.equal(withTakeaways.generalFeedback[0].summary, 'Strong session overall.')
assert.equal(withTakeaways.generalFeedback[0].photo, 'data:image/jpeg;base64,AAAA')
assert.deepEqual(withTakeaways.generalFeedback[1].takeaways, ['Solid net play', 'Move your feet'])
assert.equal(withTakeaways.generalFeedback[1].summary, '')
assert.equal(withTakeaways.timestampFeedback[0].playerPhoto, 'data:image/jpeg;base64,AAAA')

const failedAnalysis = normalizeAnalysisPreparationState({
  jobs: [{ kind: 'contacts', status: 'unexpected' }],
})
assert.equal(failedAnalysis.status, 'failed')
assert.ok(failedAnalysis.error)

// Rally-only playback: segments are second-based, padded 0.75s, merged when
// overlapping, and skip resolution jumps gaps / stops after the last rally.
const segments = buildRallySegments([
  { startFrame: 300, endFrame: 600 },   // 10s–20s
  { startFrame: 630, endFrame: 900 },   // 21s–30s — padded spans touch the first rally's, so they merge
  { startFrame: 3000, endFrame: 3300 }, // 100s–110s
  { startFrame: 'bad', endFrame: 3600 },
], 30)
assert.equal(segments.length, 2)
assert.equal(segments[0].startSeconds, 9.25)
assert.equal(segments[0].endSeconds, 30.75)
assert.equal(segments[1].startSeconds, 99.25)
assert.equal(resolveRallySkipTarget(segments, 2), 9.25)       // before first rally -> jump in
assert.equal(resolveRallySkipTarget(segments, 15), null)      // inside a rally -> play on
assert.equal(resolveRallySkipTarget(segments, 50), 99.25)     // dead time -> next rally
assert.equal(resolveRallySkipTarget(segments, 120), 'end')    // past last rally -> stop
assert.equal(resolveRallySkipTarget([], 50), null)            // no segments -> never touch playback

// Analysis countdown: heuristic total clamped to [90, 300], remaining floors at 10.
assert.equal(estimateAnalysisTotalSeconds(0), 90)
assert.equal(estimateAnalysisTotalSeconds(4), 120)
assert.equal(estimateAnalysisTotalSeconds(100), 300)
assert.equal(remainingAnalysisSeconds(120, 30), 90)
assert.equal(remainingAnalysisSeconds(120, 500), 10)
assert.equal(remainingAnalysisSeconds(null, 30), null)

console.log('PickleIQ integration fixtures passed')
