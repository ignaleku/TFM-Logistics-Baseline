import { useState, useRef, useEffect } from 'react'
import { api } from '../../api'
import type { FilesStatus, FuturePreview, PlanningProfile, RunStatus, UploadResponse } from '../../types'
import { fmtEuro, fmtPct } from '../../utils/format'

const ALL_MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

interface Props {
  filesStatus: FilesStatus | null
  onRunComplete: () => void
}

type Mode = 'historical' | 'future'

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
  const [mode, setMode] = useState<Mode>('historical')

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

  const onRunCompleteRef = useRef(onRunComplete)
  useEffect(() => { onRunCompleteRef.current = onRunComplete }, [onRunComplete])

  const [params, setParams] = useState({
    cost_late_urgent: 20,
    cost_late_normal: 5,
    worker_cost_per_hour: 15,
    hours_per_worker_month: 160,
  })

  const [selectedMonths, setSelectedMonths] = useState<string[]>([])

  // ── Future planning state ────────────────────────────────────────────────
  const [profile, setProfile] = useState<PlanningProfile | null>(null)
  const [futureMonth, setFutureMonth] = useState('December')
  const [expectedAnnual, setExpectedAnnual] = useState(240000)
  const [useOverride, setUseOverride] = useState(false)
  const [monthlyOverride, setMonthlyOverride] = useState(46000)
  const [uncertainty, setUncertainty] = useState('standard')
  const [useCurrentWorkforce, setUseCurrentWorkforce] = useState(false)
  const [currentWorkforce, setCurrentWorkforce] = useState({ picking: 2, packing: 2, dispatch: 1 })
  const [futureCosts, setFutureCosts] = useState({
    cost_late_urgent: 15, cost_late_normal: 10, worker_cost_per_hour: 18, hours_per_worker_month: 160,
  })
  const [preview, setPreview] = useState<FuturePreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  useEffect(() => {
    api.getPlanningProfile().then(setProfile).catch(() => {})
  }, [])

  const toggleMonth = (month: string) => {
    setSelectedMonths((prev) =>
      prev.includes(month) ? prev.filter((m) => m !== month) : [...prev, month]
    )
  }

  const selectAllMonths = () => setSelectedMonths([])
  const isAllMonths = selectedMonths.length === 0
  const simCount = (isAllMonths ? 12 : selectedMonths.length) * 16 * 3

  useEffect(() => {
    if (!running || startTime === null) return
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 1000)
    return () => clearInterval(id)
  }, [running, startTime])

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
      await api.runMonthlyCapacityCost({
        orders_path: 'data/uploads/orders_uploaded.csv',
        checkpoint: 'data/dqn_rl3_final.pt',
        ...params,
        months: isAllMonths ? null : selectedMonths,
      })
    } catch (e: unknown) {
      setRunError(e instanceof Error ? e.message : String(e))
      setRunning(false)
      setStartTime(null)
    }
  }

  const handlePreview = async () => {
    setPreviewLoading(true)
    setPreviewError(null)
    try {
      const result = await api.previewFuturePlan({
        planning_month: futureMonth,
        expected_annual_orders: expectedAnnual,
        monthly_orders_override: useOverride ? monthlyOverride : null,
        uncertainty_level: uncertainty,
      })
      setPreview(result)
    } catch (e: unknown) {
      setPreviewError(e instanceof Error ? e.message : String(e))
      setPreview(null)
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleFutureRun = async () => {
    setRunning(true)
    setRunError(null)
    setRunStatus(null)
    setElapsed(0)
    setStartTime(Date.now())
    try {
      await api.runFuturePlanning({
        planning_month: futureMonth,
        expected_annual_orders: expectedAnnual,
        monthly_orders_override: useOverride ? monthlyOverride : null,
        uncertainty_level: uncertainty,
        ...futureCosts,
        current_picking_workers: useCurrentWorkforce ? currentWorkforce.picking : null,
        current_packing_workers: useCurrentWorkforce ? currentWorkforce.packing : null,
        current_dispatch_workers: useCurrentWorkforce ? currentWorkforce.dispatch : null,
      })
    } catch (e: unknown) {
      setRunError(e instanceof Error ? e.message : String(e))
      setRunning(false)
      setStartTime(null)
    }
  }

  const handleLoadLatest = () => {
    onRunCompleteRef.current()
  }

  const ordersReady = uploadResult?.status === 'ok' || filesStatus?.uploaded_orders
  const hasExistingResults =
    filesStatus?.latest_recommendations_summary && filesStatus?.latest_full_results

  const progressPct = runStatus?.progress_pct ?? (running ? 10 : 0)
  const progressLabel =
    runStatus?.message ??
    (running
      ? mode === 'historical'
        ? (isAllMonths
          ? 'Running full-year monthly optimisation…'
          : `Running optimisation for: ${selectedMonths.map((m) => m.slice(0, 3)).join(', ')}…`)
        : 'Generating and optimising future scenario…'
      : '')

  const ModeTabs = (
    <div className="flex gap-1 bg-slate-100 rounded-xl p-1 w-fit">
      {(['historical', 'future'] as Mode[]).map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          disabled={running}
          className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            mode === m ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          {m === 'historical' ? 'Historical Analysis' : 'Future Planning'}
        </button>
      ))}
    </div>
  )

  const detail = runStatus?.detail
  const detailParts: string[] = []
  if (detail?.phase) detailParts.push(detail.phase)
  if (detail?.regime && detail?.regime_total) detailParts.push(`Regime ${detail.regime}/${detail.regime_total}`)
  if (detail?.finalist && detail?.finalist_total) detailParts.push(`Finalist ${detail.finalist}/${detail.finalist_total}`)
  if (detail?.replication && detail?.replication_total) detailParts.push(`Replication ${detail.replication}/${detail.replication_total}`)
  if (detail?.candidate) detailParts.push(`Candidate ${detail.candidate}`)
  if (detail?.iteration && detail?.iteration_total) detailParts.push(`Iteration ${detail.iteration}/${detail.iteration_total}`)
  if (detail?.completed_simulations != null && detail?.estimated_total_simulations != null) {
    detailParts.push(`${detail.completed_simulations}/${detail.estimated_total_simulations} simulations`)
  }

  const ProgressSection = (
    <>
      {(running || runStatus?.status === 'completed' || runStatus?.status === 'complete' || runStatus?.status === 'failed' || runStatus?.status === 'error') && (
        <div className="mt-5 space-y-3">
          <ProgressBar pct={progressPct} label={progressLabel} />
          {detailParts.length > 0 && (
            <p className="text-xs text-indigo-500">{detailParts.join(' · ')}</p>
          )}
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>{runStatus?.step ? `Step: ${runStatus.step}` : running ? 'Initialising…' : ''}</span>
            <span>Elapsed: {elapsed}s</span>
          </div>
        </div>
      )}
      {(runStatus?.status === 'completed' || runStatus?.status === 'complete') && !running && (
        <div className="mt-4 p-4 bg-emerald-50 rounded-xl border border-emerald-200">
          <p className="text-sm font-semibold text-emerald-700">✓ Simulation complete — results loaded automatically</p>
        </div>
      )}
      {runError && (
        <div className="mt-4 p-4 bg-red-50 rounded-xl border border-red-200">
          <p className="text-sm font-semibold text-red-700">Simulation failed</p>
          <pre className="text-xs text-red-600 mt-1 whitespace-pre-wrap break-all max-h-48 overflow-y-auto">{runError}</pre>
        </div>
      )}
    </>
  )

  if (mode === 'future') {
    return (
      <div className="space-y-6 max-w-3xl">
        {ModeTabs}

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-1">Future Planning</p>
          <p className="text-xs text-slate-400 mb-4">
            No per-order upload needed. The system transforms an aggregate demand forecast into simulated
            operational scenarios using the configured client planning profile for the selected month.
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1.5">Planning Month</label>
              <select className="input-field" value={futureMonth} onChange={(e) => setFutureMonth(e.target.value)}>
                {(profile?.months.map((m) => m.name) ?? ALL_MONTHS).map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1.5">Expected Annual Orders</label>
              <input
                type="number" min={1} className="input-field"
                value={expectedAnnual}
                onChange={(e) => setExpectedAnnual(parseFloat(e.target.value) || 0)}
              />
            </div>
          </div>

          <div className="mt-4">
            <label className="flex items-center gap-2 cursor-pointer select-none mb-2">
              <input
                type="checkbox" checked={useOverride}
                onChange={(e) => setUseOverride(e.target.checked)}
                className="w-4 h-4 rounded accent-indigo-600"
              />
              <span className="text-sm font-medium text-slate-700">Use monthly forecast override</span>
            </label>
            {useOverride && (
              <input
                type="number" min={1} className="input-field max-w-xs"
                value={monthlyOverride}
                onChange={(e) => setMonthlyOverride(parseFloat(e.target.value) || 0)}
              />
            )}
          </div>

          <div className="mt-4">
            <label className="block text-xs font-medium text-slate-500 mb-1.5">Uncertainty</label>
            <div className="flex gap-2">
              {(profile?.uncertainty_levels.map((u) => u.level) ?? ['low', 'standard', 'high']).map((level) => (
                <button
                  key={level}
                  onClick={() => setUncertainty(level)}
                  className={`px-4 py-1.5 rounded-full text-sm font-medium capitalize transition-colors ${
                    uncertainty === level ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-4">
            <label className="flex items-center gap-2 cursor-pointer select-none mb-2">
              <input
                type="checkbox" checked={useCurrentWorkforce}
                onChange={(e) => setUseCurrentWorkforce(e.target.checked)}
                className="w-4 h-4 rounded accent-indigo-600"
              />
              <span className="text-sm font-medium text-slate-700">Specify current workforce (optional)</span>
            </label>
            {useCurrentWorkforce && (
              <div className="grid grid-cols-3 gap-3 max-w-md">
                {(['picking', 'packing', 'dispatch'] as const).map((stage) => (
                  <div key={stage}>
                    <label className="block text-xs text-slate-400 mb-1 capitalize">{stage}</label>
                    <input
                      type="number" min={0} className="input-field"
                      value={currentWorkforce[stage]}
                      onChange={(e) => setCurrentWorkforce((w) => ({ ...w, [stage]: parseInt(e.target.value) || 0 }))}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-100">
            {[
              { key: 'cost_late_urgent', label: 'Urgent Late Cost (€/order)' },
              { key: 'cost_late_normal', label: 'Normal Late Cost (€/order)' },
              { key: 'worker_cost_per_hour', label: 'Worker Cost (€/hour)' },
              { key: 'hours_per_worker_month', label: 'Hours per Worker/Month' },
            ].map(({ key, label }) => (
              <div key={key}>
                <label className="block text-xs font-medium text-slate-500 mb-1.5">{label}</label>
                <input
                  type="number" min={0} className="input-field"
                  value={futureCosts[key as keyof typeof futureCosts]}
                  onChange={(e) => setFutureCosts((p) => ({ ...p, [key]: parseFloat(e.target.value) || 0 }))}
                />
              </div>
            ))}
          </div>

          <div className="flex gap-3 mt-5">
            <button className="btn-secondary flex-1" onClick={handlePreview} disabled={previewLoading}>
              {previewLoading ? 'Generating preview…' : 'Generate Preview'}
            </button>
            <button className="btn-primary flex-1" onClick={handleFutureRun} disabled={running || !preview}>
              {running ? 'Running…' : 'Generate & Optimise Scenario'}
            </button>
          </div>

          {previewError && <p className="text-xs text-red-600 mt-2">{previewError}</p>}
          {!preview && !previewError && (
            <p className="text-xs text-slate-400 mt-2 text-center">Generate a preview before running the scenario.</p>
          )}
        </div>

        {preview && (
          <div className="card">
            <p className="text-sm font-semibold text-slate-600 mb-1">Derived Planning Assumptions</p>
            <p className="text-xs text-slate-400 mb-4">
              These values are derived from the configured client planning profile for {preview.month_name}.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Expected Monthly Orders</p><p className="font-bold text-slate-800 text-sm">{preview.expected_monthly_orders.toLocaleString()}</p></div>
              <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Source</p><p className="font-bold text-slate-800 text-sm">{preview.source === 'monthly_override' ? 'Monthly override' : 'Annual forecast'}</p></div>
              <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Annual Share</p><p className="font-bold text-slate-800 text-sm">{fmtPct(preview.annual_share)}</p></div>
              <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Urgent Share</p><p className="font-bold text-slate-800 text-sm">{fmtPct(preview.urgent_share)}</p></div>
              <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Avg Items/Order</p><p className="font-bold text-slate-800 text-sm">{preview.expected_avg_items.toFixed(1)}</p></div>
              <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Operating Days</p><p className="font-bold text-slate-800 text-sm">{preview.operating_days}</p></div>
              <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Orders / Operating Hour</p><p className="font-bold text-slate-800 text-sm">{preview.expected_orders_per_operating_hour}</p></div>
              <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Replications</p><p className="font-bold text-slate-800 text-sm">{preview.replications}</p></div>
            </div>
            <div className="grid grid-cols-2 gap-4 mt-3">
              <div className="bg-indigo-50 rounded-xl p-3 text-xs">
                <p className="text-indigo-400 mb-1">Product Family Mix</p>
                {Object.entries(preview.product_family_shares).map(([k, v]) => (
                  <div key={k} className="flex justify-between"><span className="capitalize text-slate-600">{k}</span><strong>{fmtPct(v)}</strong></div>
                ))}
              </div>
              <div className="bg-indigo-50 rounded-xl p-3 text-xs">
                <p className="text-indigo-400 mb-1">Complexity Mix</p>
                {Object.entries(preview.complexity_shares).map(([k, v]) => (
                  <div key={k} className="flex justify-between"><span className="capitalize text-slate-600">{k}</span><strong>{fmtPct(v)}</strong></div>
                ))}
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-3">
              Uncertainty ({preview.uncertainty_level}): demand CV {fmtPct(preview.uncertainty_assumptions.demand_cv)},
              arrival CV {fmtPct(preview.uncertainty_assumptions.arrival_cv)}. SLA targets: urgent ≥{fmtPct(preview.sla_targets.urgent_target, 0)},
              normal ≥{fmtPct(preview.sla_targets.normal_target, 0)}.
            </p>
          </div>
        )}

        <div className="card">
          {ProgressSection}
          {!running && !runStatus && (
            <p className="text-xs text-slate-400 text-center">
              Results are scenario-based estimates over simulated replications, not guarantees.
            </p>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-3xl">
      {ModeTabs}

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
            ? `All 12 months selected — ${12 * 16 * 3} simulations`
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
          {' '}× 16 worker regimes × 3 policies (FIFO, Urgent-First, RL-3 DQN) —{' '}
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

        {ProgressSection}
      </div>
    </div>
  )
}
