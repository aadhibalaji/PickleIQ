import React, { useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import ScoreboardPanel from './ScoreboardPanel.jsx'

export default function ReportPanel({ videoId, onRegenerate }) {
  const [report, setReport] = useState(null)
  const [history, setHistory] = useState([])
  const [params, setParams] = useState(null)
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState('')
  const timerRef = useRef(null)

  const refresh = () => {
    api.listReports(videoId).then(setHistory).catch(() => {})
    api.getParams().then(setParams).catch(() => {})
  }
  useEffect(() => { refresh() }, [videoId]) // eslint-disable-line
  useEffect(() => () => clearInterval(timerRef.current), [])

  const generate = async () => {
    setBusy(true)
    setError('')
    setElapsed(0)
    const t0 = Date.now()
    timerRef.current = setInterval(() => setElapsed((Date.now() - t0) / 1000), 500)
    try {
      // Single long request: metrics are instant, the Claude analysis is
      // most of the wait (~30-90 s).
      setReport(await api.generateReport(videoId, 'pass2'))
      refresh()
    } catch (e) { setError(e.message) }
    clearInterval(timerRef.current)
    setBusy(false)
  }

  const stage = elapsed < 3 ? 'Comparing your labels against Pass 2…'
    : elapsed < 8 ? 'Extracting feature traces around each disputed boundary…'
    : 'Asking Claude to analyze the discrepancies… (typically 30–90 s)'

  const applySuggestions = async (suggestions) => {
    const r = await api.applySuggestions(report.report_id, suggestions)
    setError('')
    refresh()
    setReport({ ...report, applied_llm_version: r.applied_version })
  }

  const s = report?.summary
  return (
    <div>
      <div className="row spread">
        <div className="muted">
          Compares your labels against Pass 2, saves discrepancy records + feature traces,
          applies deterministic global parameter corrections, and asks the Anthropic API to
          explain why the discrepancies happened. Improvements apply to <b>every future clip</b>.
        </div>
        <div className="row">
          <button className="primary" onClick={generate} disabled={busy}>
            {busy ? 'Analyzing…' : 'Generate report'}
          </button>
          <button onClick={() => onRegenerate('pass2')} disabled={busy}>Regenerate video (Pass 2)</button>
          <button onClick={() => onRegenerate('pass1')} disabled={busy}>Regenerate all (Pass 1 + 2)</button>
        </div>
      </div>
      {busy && (
        <div className="row" style={{ margin: '10px 0' }}>
          <div className="loadingbar" />
          <span className="muted">{stage} {elapsed.toFixed(0)}s</span>
        </div>
      )}
      {error && <div className="error">{error}</div>}

      {report && (
        <>
          <h2>Metrics</h2>
          <table>
            <tbody>
              <tr>
                <td>Human rallies: <b>{s.human_rallies}</b></td>
                <td>AI rallies: <b>{s.ai_rallies}</b></td>
                <td>Matched: <b>{s.matched}</b></td>
                <td>Missed: <b className={s.missed ? 'error' : ''}>{s.missed}</b></td>
                <td>Extra: <b className={s.extra ? 'error' : ''}>{s.extra}</b></td>
                <td>Mean IoU: <b>{s.mean_iou ?? '—'}</b></td>
                <td>Median start err: <b>{s.median_start_error_frames ?? '—'}f</b></td>
                <td>Median end err: <b>{s.median_end_error_frames ?? '—'}f</b></td>
              </tr>
            </tbody>
          </table>

          <h2>Discrepancies</h2>
          <table>
            <thead><tr><th>Type</th><th>Human</th><th>AI</th><th>IoU</th><th>Start err</th><th>End err</th></tr></thead>
            <tbody>
              {report.discrepancies.map((d, i) => (
                <tr key={i}>
                  <td><span className={`pill ${d.type === 'boundary_drift' ? 'drift' : d.type}`}>{d.type}</span></td>
                  <td>{d.human_rally ? `${d.human_rally.start_frame} → ${d.human_rally.end_frame}` : '—'}</td>
                  <td>{d.ai_rally ? `${d.ai_rally.start_frame} → ${d.ai_rally.end_frame}` : '—'}</td>
                  <td>{d.temporal_iou}</td>
                  <td>{d.start_error != null ? `${d.start_error}f (${d.start_error_ms}ms)` : '—'}</td>
                  <td>{d.end_error != null ? `${d.end_error}f (${d.end_error_ms}ms)` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {report.deterministic_rationale?.length > 0 && (
            <>
              <h2>Deterministic parameter updates (applied globally{report.applied_params_version ? ` — v${report.applied_params_version}` : ''})</h2>
              <ul>{report.deterministic_rationale.map((r, i) => <li key={i}>{r}</li>)}</ul>
            </>
          )}

          <h2>LLM analysis</h2>
          {report.llm_report
            ? <div className="mdreport">{report.llm_report}</div>
            : <div className="muted">No LLM report — set ANTHROPIC_API_KEY in .env to enable the feedback analysis.</div>}

          {report.llm_suggestions?.length > 0 && (
            <>
              <h2>LLM parameter suggestions</h2>
              <table>
                <thead><tr><th>Param</th><th>Value</th><th>Scope</th><th>Rationale</th></tr></thead>
                <tbody>
                  {report.llm_suggestions.map((sg, i) => (
                    <tr key={i}>
                      <td>{sg.param}</td><td>{sg.value}</td>
                      <td><span className={`pill ${sg.scope === 'global' ? 'human' : ''}`}>{sg.scope}</span></td>
                      <td className="muted">{sg.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="row" style={{ marginTop: 6 }}>
                <button className="blue" onClick={() => applySuggestions(report.llm_suggestions)}>
                  Apply global suggestions{report.applied_llm_version ? ` (applied → v${report.applied_llm_version})` : ''}
                </button>
                <span className="muted">clip-local suggestions are never written to the global baseline</span>
              </div>
            </>
          )}
        </>
      )}

      {params && (
        <>
          <h2>Learned parameters — global baseline (v{params.version})</h2>
          <div className="muted" style={{ marginBottom: 6 }}>
            Every new upload loads this version before Pass 1 runs — accumulated lessons apply automatically.
          </div>
          <table>
            <thead><tr><th>Parameter</th><th>Current</th><th>Default</th></tr></thead>
            <tbody>
              {Object.entries(params.params).map(([k, v]) => (
                <tr key={k} style={v !== params.defaults[k] ? { color: '#4cc38a' } : {}}>
                  <td>{k}</td><td>{typeof v === 'number' ? +v.toFixed(3) : String(v)}</td>
                  <td className="muted">{String(params.defaults[k])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <ScoreboardPanel refreshKey={`${videoId}:${params?.version || ''}:${report?.applied_params_version || ''}`} />

      {history.length > 0 && (
        <>
          <h2>Past reports</h2>
          <table>
            <thead><tr><th>Date</th><th>IoU</th><th>Missed</th><th>Extra</th><th>Params version created</th></tr></thead>
            <tbody>
              {history.map((r) => (
                <tr key={r.id}>
                  <td>{new Date(r.created_at * 1000).toLocaleString()}</td>
                  <td>{r.summary.mean_iou ?? '—'}</td>
                  <td>{r.summary.missed}</td>
                  <td>{r.summary.extra}</td>
                  <td>{r.applied_params_version ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}
