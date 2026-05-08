import { useState, useRef } from 'react'
import { api } from '../../api'
import type { FilesStatus, RunResponse, UploadResponse } from '../../types'
import { fmtEuro } from '../../utils/format'

interface Props {
  filesStatus: FilesStatus | null
  onRunComplete: () => void
}

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-block w-2 h-2 rounded-full ${ok ? 'bg-emerald-400' : 'bg-slate-300'}`} />
  )
}

export function UploadRunTab({ filesStatus, onRunComplete }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  const [runResult, setRunResult] = useState<RunResponse | null>(null)
  const [runError, setRunError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  const [params, setParams] = useState({
    cost_late_urgent: 20,
    cost_late_normal: 5,
    worker_cost_per_hour: 15,
    hours_per_worker_month: 160,
  })

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
    setRunResult(null)
    try {
      const result = await api.runMonthlyCapacityCost({
        orders_path: 'data/uploads/orders_uploaded.csv',
        checkpoint: 'data/dqn_rl5_v2_final.pt',
        ...params,
      })
      setRunResult(result)
      onRunComplete()
    } catch (e: unknown) {
      setRunError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }

  const ordersReady = uploadResult?.status === 'ok' || filesStatus?.uploaded_orders

  return (
    <div className="space-y-6 max-w-3xl">
      {/* System status */}
      {filesStatus && (
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">System Status</p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { label: 'Orders Uploaded', ok: filesStatus.uploaded_orders },
              { label: 'RL-5 Checkpoint', ok: filesStatus.checkpoint },
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
            Estimated monthly labour cost at 7 workers:{' '}
            <strong>{fmtEuro(7 * params.worker_cost_per_hour * params.hours_per_worker_month)}</strong>
          </p>
        </div>
      </div>

      {/* Run */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-2">3. Run Monthly Optimisation</p>
        <p className="text-xs text-slate-400 mb-4">
          Evaluates all months × all worker regimes × 3 policies. This may take several minutes.
        </p>

        {!filesStatus?.checkpoint && (
          <div className="mb-4 p-3 bg-amber-50 rounded-xl border border-amber-200">
            <p className="text-xs text-amber-700">⚠ RL-5 checkpoint not found. Ensure <code>data/dqn_rl5_v2_final.pt</code> exists on the backend.</p>
          </div>
        )}

        <button
          className="btn-primary w-full"
          onClick={handleRun}
          disabled={running || !ordersReady}
        >
          {running ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
              </svg>
              Running optimisation…
            </span>
          ) : 'Run Monthly Optimisation'}
        </button>

        {!ordersReady && (
          <p className="text-xs text-slate-400 text-center mt-2">Upload an orders CSV first.</p>
        )}

        {runResult && (
          <div className="mt-4 p-4 bg-emerald-50 rounded-xl border border-emerald-200">
            <p className="text-sm font-semibold text-emerald-700 mb-2">
              ✓ Simulation complete — {runResult.elapsed_seconds}s
            </p>
            <p className="text-xs text-slate-500 mb-1">Run ID: <code>{runResult.run_id}</code></p>
            <div className="space-y-1">
              {Object.entries(runResult.output_paths).map(([k, v]) => (
                <p key={k} className="text-xs text-slate-600">
                  <span className="font-medium">{k}:</span> <code>{v}</code>
                </p>
              ))}
            </div>
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
