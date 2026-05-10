import { useState, useRef, useEffect } from 'react'
import { api } from '../../api'
import type { FilesStatus, RunStatus, UploadResponse } from '../../types'
import { fmtEuro } from '../../utils/format'

const ALL_MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

interface Props {
  filesStatus: FilesStatus | null
  onRunComplete: () => void
}

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-block w-2 h-2 rounded-full ${ok ? 'bg-emerald-400' : 'bg-slate-300'}`} />
  )
}

function ProgressBar({ pct, label }: { pct: number; label: string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-slate-500">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
        <div
          className="h-2 rounded-full bg-indigo-500 transition-all duration-700 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export function UploadRunTab({ filesStatus, onRunComplete }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  const [runError, setRunError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  // Progress state
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [startTime, setStartTime] = useState<number | null>(null)

  // Keep a stable ref to onRunComplete to avoid stale closures in the polling effect
  const onRunCompleteRef = useRef(onRunComplete)
  useEffect(() => { onRunCompleteRef.current = onRunComplete }, [onRunComplete])

  const [params, setParams] = useState({
    cost_late_urgent: 20,
    cost_late_normal: 5,
    worker_cost_per_hour: 15,
    hours_per_worker_month: 160,
  })

  // Empty array = all months selected (sends null to API)
  const [selectedMonths, setSelectedMonths] = useState<string[]>([])

  const toggleMonth = (month: string) => {
    setSelectedMonths((prev) =>
      prev.includes(month) ? prev.filter((m) => m !== month) : [...prev, month]
    )
  }

  const selectAllMonths = () => setSelectedMonths([])
  const isAllMonths = selectedMonths.length === 0
  const simCount = (isAllMonths ? 12 : selectedMonths.length) * 7 * 3

  // Elapsed timer
  useEffect(() => {
    if (!running || startTime === null) return
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 1000)
    return () => clearInterval(id)
  }, [running, startTime])

  // Poll /run/status while running; detect completion/error and fire onRunComplete
  useEffect(() => {
    if (!running) return
    const id = setInterval(async () => {
      try {
        const status = await api.runStatus()
        setRunStatus(status)
        const isDone = status.status === 'completed' || status.status === 'complete'
        const isFailed = status.status === 'failed' || status.status === 'error'
        if (isDone) {
          setRunning(false)
          setStartTime(null)
          onRunCompleteRef.current()
        } else if (isFailed) {
          setRunning(false)
          setStartTime(null)
          setRunError(status.error ?? status.message ?? 'Simulation failed on the backend.')
        }
      } catch {
        // backend busy or transient network issue — keep polling
      }
    }, 2000)
    return () => clearInterval(id)
  }, [running])

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0]
    if (!file) return
    setUploading(true)
    setUploadError(null)
    setUploadResult(null)
    try {
      const result = await api.uploadOrders(file)
      setUploadResult(result)
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
    }
  }

  const handleRun = async () => {
    setRunning(true)
    setRunError(null)
    setRunStatus(null)
    setElapsed(0)
    setStartTime(Date.now())
    try {
      // POST returns immediately with {status: "started"}
      // The polling effect above detects completion and calls onRunComplete
      await api.runMonthlyCapacityCost({
        orders_path: 'data/uploads/orders_uploaded.csv',
        checkpoint: 'data/dqn_rl3_final.pt',
        ...params,
        months: isAllMonths ? null : selectedMonths,
      })
    } catch (e: unknown) {
      // 409 (already running) or validation errors come through here
      setRunError(e instanceof Error ? e.message : String(e))
      setRunning(false)
      setStartTime(null)
    }
  }

  const handleLoadLatest = () => {
    onRunCompleteRef.current()
  }

  const ordersReady = uploadResult?.status === 'ok' || filesStatus?.uploaded_orders
  // Enable "Load Latest Results" only when the final export files exist (not just the raw capacity CSV)
  const hasExistingResults =
    filesStatus?.latest_recommendations_summary && filesStatus?.latest_full_results

  const progressPct = runStatus?.progress_pct ?? (running ? 10 : 0)
  const progressLabel =
    runStatus?.message ??
    (running
      ? isAllMonths
        ? 'Running full-year monthly optimisation…'
        : `Running optimisation for: ${selectedMonths.map((m) => m.slice(0, 3)).join(', ')}…`
      : '')

  return (
    <div className="space-y-6 max-w-3xl">
      {/* System status */}
      {filesStatus && (
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">System Status</p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { label: 'Orders Uploaded', ok: filesStatus.uploaded_orders },
              { label: 'RL-3 Checkpoint', ok: filesStatus.checkpoint },
              { label: 'Capacity Results', ok: filesStatus.latest_capacity_results },
              { label: 'Recommendations', ok: filesStatus.latest_recommendations_summary },
              { label: 'Full Results', ok: filesStatus.latest_full_results },
            ].map((s) => (
              <div key={s.label} className="flex items-center gap-2 text-sm">
                <StatusDot ok={s.ok} />
                <span className={s.ok ? 'text-slate-700' : 'text-slate-400'}>{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-4">1. Upload Historical Orders CSV</p>
        <p className="text-xs text-slate-400 mb-4">
          Required columns: <code>order_id, arrival_time, order_type, num_items, product_class</code>
        </p>

        <div className="flex gap-3 items-center">
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="text-sm text-slate-500 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-medium file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer"
          />
          <button className="btn-primary" onClick={handleUpload} disabled={uploading}>
            {uploading ? 'Uploading…' : 'Upload'}
          </button>
        </div>

        {uploadResult && (
          <div className="mt-4 p-4 bg-emerald-50 rounded-xl border border-emerald-200">
            <p className="text-sm font-semibold text-emerald-700 mb-2">✓ {uploadResult.message}</p>
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div><span className="text-slate-500">Rows:</span> <strong>{uploadResult.total_rows.toLocaleString()}</strong></div>
              <div><span className="text-slate-500">Date range:</span> <strong>{uploadResult.date_range ?? '—'}</strong></div>
              <div><span className="text-slate-500">Urgent share:</span> <strong>{((uploadResult.urgent_share ?? 0) * 100).toFixed(1)}%</strong></div>
            </div>
            {uploadResult.detected_months.length > 0 && (
              <p className="text-xs text-slate-500 mt-2">
                Months: {uploadResult.detected_months.join(', ')}
              </p>
            )}
          </div>
        )}

        {uploadError && (
          <div className="mt-4 p-4 bg-red-50 rounded-xl border border-red-200">
            <p className="text-sm font-semibold text-red-700">Upload failed</p>
            <p className="text-xs text-red-600 mt-1">{uploadError}</p>
          </div>
        )}
      </div>

      {/* Economic params */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-4">2. Economic Assumptions</p>
        <div className="grid grid-cols-2 gap-4">
          {[
            { key: 'cost_late_urgent', label: 'Urgent Late Cost (€/order)', min: 0, step: 1 },
            { key: 'cost_late_normal', label: 'Normal Late Cost (€/order)', min: 0, step: 1 },
            { key: 'worker_cost_per_hour', label: 'Worker Cost (€/hour)', min: 0, step: 0.5 },
            { key: 'hours_per_worker_month', label: 'Hours per Worker/Month', min: 1, step: 8 },
          ].map(({ key, label, min, step }) => (
            <div key={key}>
              <label className="block text-xs font-medium text-slate-500 mb-1.5">{label}</label>
              <input
                type="number"
                min={min}
                step={step}
                value={params[key as keyof typeof params]}
                onChange={(e) => setParams((p) => ({ ...p, [key]: parseFloat(e.target.value) || 0 }))}
                className="input-field"
              />
            </div>
          ))}
        </div>
        <div className="mt-4 p-3 bg-slate-50 rounded-xl">
          <p className="text-xs text-slate-500">
            Estimated monthly labour cost at 6 workers:{' '}
            <strong>{fmtEuro(6 * params.worker_cost_per_hour * params.hours_per_worker_month)}</strong>
          </p>
        </div>
      </div>

      {/* Month selector */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-1">3. Months to Simulate</p>
        <p className="text-xs text-slate-400 mb-4">
          Running fewer months is useful for interactive analysis. Full-year optimisation evaluates all 12 months.
        </p>

        <div className="flex items-center gap-3 mb-4">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={isAllMonths}
              onChange={selectAllMonths}
              className="w-4 h-4 rounded accent-indigo-600"
            />
            <span className="text-sm font-medium text-slate-700">All months</span>
          </label>
          {!isAllMonths && (
            <button
              className="text-xs text-indigo-500 hover:text-indigo-700 underline"
              onClick={selectAllMonths}
            >
              Clear selection
            </button>
          )}
        </div>

        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
          {ALL_MONTHS.map((month) => {
            const checked = !isAllMonths && selectedMonths.includes(month)
            return (
              <label
                key={month}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl border cursor-pointer select-none text-sm transition-colors ${
                  checked
                    ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                    : isAllMonths
                    ? 'border-slate-200 bg-slate-50 text-slate-500'
                    : 'border-slate-200 bg-white text-slate-600 hover:border-indigo-300'
                }`}
              >
                <input
                  type="checkbox"
                  checked={isAllMonths || checked}
                  onChange={() => {
                    if (isAllMonths) {
                      // First click deselects all → selects only this month
                      setSelectedMonths(ALL_MONTHS.filter((m) => m !== month))
                    } else {
                      toggleMonth(month)
                    }
                  }}
                  className="w-3.5 h-3.5 rounded accent-indigo-600"
                />
                {month.slice(0, 3)}
              </label>
            )
          })}
        </div>

        <p className="text-xs text-slate-400 mt-3">
          {isAllMonths
            ? `All 12 months selected — ${12 * 7 * 3} simulations`
            : selectedMonths.length === 0
            ? 'No months selected'
            : `${selectedMonths.length} month${selectedMonths.length > 1 ? 's' : ''} selected — ${simCount} simulations`}
        </p>
      </div>

      {/* Run */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-2">4. Run Monthly Optimisation</p>
        <p className="text-xs text-slate-400 mb-4">
          Evaluates{' '}
          {isAllMonths ? 'all 12 months' : `${selectedMonths.length} selected month${selectedMonths.length !== 1 ? 's' : ''}`}
          {' '}× 7 worker regimes × 3 policies (FIFO, Urgent-First, RL-3 DQN) —{' '}
          {simCount} simulations total.
          The simulation runs in the background — you can track progress below.
        </p>

        {!filesStatus?.checkpoint && (
          <div className="mb-4 p-3 bg-amber-50 rounded-xl border border-amber-200">
            <p className="text-xs text-amber-700">
              ⚠ RL-3 checkpoint not found. Ensure <code>data/dqn_rl3_final.pt</code> exists on the backend.
            </p>
          </div>
        )}

        <div className="flex gap-3">
          <button
            className="btn-primary flex-1"
            onClick={handleRun}
            disabled={running || !ordersReady || (!isAllMonths && selectedMonths.length === 0)}
          >
            {running ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
                Running in background…
              </span>
            ) : 'Run Monthly Optimisation'}
          </button>

          <button
            className="btn-secondary"
            onClick={handleLoadLatest}
            disabled={running || !hasExistingResults}
            title={hasExistingResults ? 'Load already-generated results without re-running' : 'No results available yet'}
          >
            Load Latest Results
          </button>
        </div>

        {!ordersReady && (
          <p className="text-xs text-slate-400 text-center mt-2">Upload an orders CSV first.</p>
        )}
        {ordersReady && !isAllMonths && selectedMonths.length === 0 && (
          <p className="text-xs text-slate-400 text-center mt-2">Select at least one month above.</p>
        )}

        {/* Progress bar */}
        {(running || runStatus?.status === 'completed' || runStatus?.status === 'complete' || runStatus?.status === 'failed' || runStatus?.status === 'error') && (
          <div className="mt-5 space-y-3">
            <ProgressBar pct={progressPct} label={progressLabel} />
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>
                {runStatus?.step ? `Step: ${runStatus.step}` : running ? 'Initialising…' : ''}
              </span>
              <span>Elapsed: {elapsed}s</span>
            </div>
          </div>
        )}

        {(runStatus?.status === 'completed' || runStatus?.status === 'complete') && !running && (
          <div className="mt-4 p-4 bg-emerald-50 rounded-xl border border-emerald-200">
            <p className="text-sm font-semibold text-emerald-700">
              ✓ Simulation complete — results loaded automatically
            </p>
          </div>
        )}

        {runError && (
          <div className="mt-4 p-4 bg-red-50 rounded-xl border border-red-200">
            <p className="text-sm font-semibold text-red-700">Simulation failed</p>
            <pre className="text-xs text-red-600 mt-1 whitespace-pre-wrap break-all max-h-48 overflow-y-auto">{runError}</pre>
          </div>
        )}
      </div>
    </div>
  )
}
