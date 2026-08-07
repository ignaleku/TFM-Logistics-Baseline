import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer,
} from 'recharts'
import { api } from '../../api'
import type { OrderSummary, RunScope, RunScopeOrderSummaryRow } from '../../types'
import { fmtPct } from '../../utils/format'

const MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

// Backend convention: shares are proportions in [0, 1]. Only multiply by 100 for display here.
// Reject non-finite/missing values explicitly rather than letting NaN reach a chart.
function safePct(value: number | null | undefined): number {
  const n = Number(value ?? 0)
  return Number.isFinite(n) ? +(n * 100).toFixed(1) : 0
}

function safeNum(value: number | null | undefined): number {
  const n = Number(value ?? 0)
  return Number.isFinite(n) ? n : 0
}

function validRows(rows: RunScopeOrderSummaryRow[] | OrderSummary[]): RunScopeOrderSummaryRow[] {
  return (rows as RunScopeOrderSummaryRow[]).filter((d) => {
    const famSum = safeNum(d.pct_standard) + safeNum(d.pct_fragile) + safeNum(d.pct_bulky)
    const cplSum = safeNum(d.pct_low) + safeNum(d.pct_medium) + safeNum(d.pct_high)
    return famSum > 0.9 && famSum < 1.1 && cplSum > 0.9 && cplSum < 1.1
  })
}

function Kpi({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card text-center py-4">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className="text-2xl font-bold text-slate-800">{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
    </div>
  )
}

function ScopedCharts({ rows, title }: { rows: RunScopeOrderSummaryRow[]; title: string }) {
  const rowsOk = validRows(rows)
  if (!rowsOk.length) {
    return (
      <div className="p-4 bg-amber-50 rounded-xl border border-amber-200 text-sm text-amber-800">
        Product-family / complexity shares for {title.toLowerCase()} are not available or do not sum to ~100% —
        skipping the mix charts rather than showing misleading values.
      </div>
    )
  }

  const familyData = rowsOk.map((d) => ({
    month: MONTH_ABBR[d.month - 1],
    standard: safePct(d.pct_standard),
    fragile: safePct(d.pct_fragile),
    bulky: safePct(d.pct_bulky),
  }))

  const complexityData = rowsOk.map((d) => ({
    month: MONTH_ABBR[d.month - 1],
    low: safePct(d.pct_low),
    medium: safePct(d.pct_medium),
    high: safePct(d.pct_high),
  }))

  const workloadData = rowsOk.map((d) => ({
    month: MONTH_ABBR[d.month - 1],
    picking: safeNum(d.avg_picking_units),
    packing: safeNum(d.avg_packing_units),
    dispatch: safeNum(d.avg_dispatch_units),
  }))

  return (
    <>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-1">Product Family Mix (%)</p>
          <p className="text-xs text-slate-400 mb-4">
            Fragile orders need 1.8× more packing time. Bulky orders need 1.6×.
          </p>
          <ResponsiveContainer width="100%" height={rowsOk.length > 1 ? 240 : 180}>
            <BarChart data={familyData} layout={rowsOk.length > 1 ? 'horizontal' : 'vertical'}
              margin={{ top: 4, right: 16, bottom: 4, left: rowsOk.length > 1 ? 8 : 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              {rowsOk.length > 1 ? (
                <>
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10 }} />
                </>
              ) : (
                <>
                  <XAxis type="number" tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
                  <YAxis dataKey="month" type="category" tick={{ fontSize: 11 }} width={40} />
                </>
              )}
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="standard" name="Standard" fill="#6366f1" stackId="a" />
              <Bar dataKey="fragile" name="Fragile" fill="#f97316" stackId="a" />
              <Bar dataKey="bulky" name="Bulky" fill="#0ea5e9" stackId="a" radius={rowsOk.length > 1 ? [4, 4, 0, 0] : [0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-1">Complexity Mix (%)</p>
          <p className="text-xs text-slate-400 mb-4">
            High-complexity orders need up to 1.7× more packing and 1.4× more picking time.
          </p>
          <ResponsiveContainer width="100%" height={rowsOk.length > 1 ? 240 : 180}>
            <BarChart data={complexityData} layout={rowsOk.length > 1 ? 'horizontal' : 'vertical'}
              margin={{ top: 4, right: 16, bottom: 4, left: rowsOk.length > 1 ? 8 : 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              {rowsOk.length > 1 ? (
                <>
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10 }} />
                </>
              ) : (
                <>
                  <XAxis type="number" tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
                  <YAxis dataKey="month" type="category" tick={{ fontSize: 11 }} width={40} />
                </>
              )}
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="low" name="Low" fill="#34d399" stackId="a" />
              <Bar dataKey="medium" name="Medium" fill="#fbbf24" stackId="a" />
              <Bar dataKey="high" name="High" fill="#f87171" stackId="a" radius={rowsOk.length > 1 ? [4, 4, 0, 0] : [0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-1">Expected Workload by Operation</p>
        <p className="text-xs text-slate-400 mb-4">
          Average workload units per order at each stage — the operational load indicator passed to the
          simulation. Packing units are the most variable, driven by product_family × complexity_level.
        </p>
        <ResponsiveContainer width="100%" height={rowsOk.length > 1 ? 240 : 200}>
          {rowsOk.length > 1 ? (
            <LineChart data={workloadData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => v.toFixed(2)} />
              <Legend />
              <Line type="monotone" dataKey="picking" name="Picking units" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="packing" name="Packing units" stroke="#f97316" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="dispatch" name="Dispatch units" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          ) : (
            <BarChart data={workloadData} layout="vertical" margin={{ top: 4, right: 32, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="month" type="category" tick={{ fontSize: 11 }} width={40} />
              <Tooltip formatter={(v: number) => v.toFixed(2)} />
              <Legend />
              <Bar dataKey="picking" name="Picking units" fill="#6366f1" radius={[0, 4, 4, 0]} />
              <Bar dataKey="packing" name="Packing units" fill="#f97316" radius={[0, 4, 4, 0]} />
              <Bar dataKey="dispatch" name="Dispatch units" fill="#10b981" radius={[0, 4, 4, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </>
  )
}

function AnnualProfile() {
  const [data, setData] = useState<OrderSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.getOrderSummary()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <p className="text-sm text-slate-400 py-6 text-center">Loading annual client profile…</p>
  if (error) return (
    <div className="p-4 bg-red-50 rounded-xl border border-red-200 text-sm text-red-700">
      <strong>Could not load annual profile.</strong> {error}
    </div>
  )
  if (!data.length) return <p className="text-sm text-slate-400 py-6 text-center">No annual baseline data found.</p>

  const rowsOk = validRows(data)
  const demandData = rowsOk.map((d) => ({
    month: MONTH_ABBR[d.month - 1],
    orders: safeNum(d.orders),
    urgentShare: safePct(d.urgent_share),
  }))

  return (
    <div className="space-y-6 mt-2">
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Monthly Order Volume (annual baseline)</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={demandData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => (v / 1000).toFixed(0) + 'k'} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: number) => v.toLocaleString()} />
              <Bar dataKey="orders" name="Orders" fill="#6366f1" radius={[4, 4, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Urgent Order Share by Month (%)</p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={demandData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => `${v}%`} domain={[0, 35]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => `${v}%`} />
              <Line type="monotone" dataKey="urgentShare" name="Urgent %" stroke="#f97316" strokeWidth={2.5} dot={{ r: 4, fill: '#f97316' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <ScopedCharts rows={rowsOk} title="the annual baseline" />
    </div>
  )
}

export function DemandComplexityTab() {
  const [scope, setScope] = useState<RunScope | null>(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAnnual, setShowAnnual] = useState(false)

  useEffect(() => {
    api.getLatestRunScope()
      .then(setScope)
      .catch((e) => {
        const msg = e instanceof Error ? e.message : String(e)
        if (/No run scope/i.test(msg)) setNotFound(true)
        else setError(msg)
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-center py-24 text-slate-400">Loading run scope…</div>

  if (error) return (
    <div className="p-4 bg-red-50 rounded-xl border border-red-200 text-sm text-red-700">
      <strong>Could not load demand &amp; complexity for this run.</strong> {error}
    </div>
  )

  if (notFound || !scope) {
    return (
      <div className="space-y-6">
        <div className="text-center py-16 text-slate-400">
          Run an analysis to view demand and complexity for the selected scope.
        </div>
        <div className="text-center">
          <button className="text-xs text-indigo-500 hover:text-indigo-700 underline" onClick={() => setShowAnnual((v) => !v)}>
            {showAnnual ? 'Hide' : 'View'} annual client profile
          </button>
        </div>
        {showAnnual && <AnnualProfile />}
      </div>
    )
  }

  const rows = scope.order_summary ?? []
  const isFuture = scope.run_mode === 'future'
  const preview = scope.preview

  return (
    <div className="space-y-8">
      {/* Explanation banner */}
      <div className="p-4 bg-indigo-50 rounded-xl border border-indigo-100 text-sm text-indigo-800">
        <strong>Why this matters:</strong> Orders are not homogeneous. A <em>fragile, high-complexity</em> order
        requires 3× more packing time than a <em>standard, low-complexity</em> one — even with the same item count.
        This creates shifting bottlenecks across Picking, Packing, and Dispatch, making intelligent sequencing (RL-3)
        more valuable than a fixed policy.
      </div>

      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-sm font-semibold text-slate-600">
          {isFuture
            ? `Future Planning scenario — ${scope.month_name ?? scope.month_names?.[0] ?? ''}`
            : `Historical Analysis — ${scope.month_names?.join(', ') || 'scoped months'}`}
        </p>
        <span className="text-xs px-2.5 py-1 rounded-full bg-slate-100 text-slate-500 font-medium">
          {isFuture ? 'Simulated scenario, replication #1' : `${scope.months?.length ?? rows.length} month(s) in this run`}
        </span>
      </div>

      {/* A. Scenario summary (future) */}
      {isFuture && preview && (
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-1">Scenario Summary</p>
          <p className="text-xs text-slate-400 mb-4">
            Derived planning assumptions for {preview.month_name}, from the configured client planning profile.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Expected Monthly Orders</p><p className="font-bold text-slate-800 text-sm">{preview.expected_monthly_orders.toLocaleString()}</p></div>
            <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Forecast Source</p><p className="font-bold text-slate-800 text-sm">{preview.source === 'monthly_override' ? 'Monthly override' : 'Annual forecast'}</p></div>
            <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Annual Share</p><p className="font-bold text-slate-800 text-sm">{fmtPct(preview.annual_share)}</p></div>
            <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Urgent Share</p><p className="font-bold text-slate-800 text-sm">{fmtPct(preview.urgent_share)}</p></div>
            <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Avg Items/Order</p><p className="font-bold text-slate-800 text-sm">{preview.expected_avg_items.toFixed(1)}</p></div>
            <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Operating Days</p><p className="font-bold text-slate-800 text-sm">{preview.operating_days}</p></div>
            <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Orders / Operating Hour</p><p className="font-bold text-slate-800 text-sm">{preview.expected_orders_per_operating_hour}</p></div>
            <div className="bg-slate-50 rounded-xl p-3"><p className="text-slate-400">Replications</p><p className="font-bold text-slate-800 text-sm">{preview.replications}</p></div>
          </div>
          <p className="text-xs text-slate-400 mt-3">
            Uncertainty ({preview.uncertainty_level}): demand CV {fmtPct(preview.uncertainty_assumptions.demand_cv)},
            arrival CV {fmtPct(preview.uncertainty_assumptions.arrival_cv)}.
          </p>
        </div>
      )}

      {/* KPIs (historical, or future without preview) */}
      {!isFuture && rows.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Kpi label="Total orders (scoped)" value={rows.reduce((s, d) => s + safeNum(d.orders), 0).toLocaleString()} />
          <Kpi label="Avg urgent share" value={`${(rows.reduce((s, d) => s + safeNum(d.urgent_share), 0) / rows.length * 100).toFixed(1)}%`} />
          <Kpi label="Months in run" value={String(rows.length)} sub={scope.month_names?.join(', ')} />
          <Kpi label="Avg items/order" value={(rows.reduce((s, d) => s + safeNum(d.mean_num_items), 0) / rows.length).toFixed(1)} />
        </div>
      )}

      {/* B/C/D — family mix, complexity mix, workload by operation, scoped to the run */}
      <ScopedCharts rows={rows} title={isFuture ? (scope.month_name ?? 'this scenario') : 'this run'} />

      {/* E. Optional annual profile */}
      <div className="pt-2 border-t border-slate-100">
        <button className="text-xs text-indigo-500 hover:text-indigo-700 underline" onClick={() => setShowAnnual((v) => !v)}>
          {showAnnual ? 'Hide' : 'View'} annual client profile
        </button>
        {showAnnual && <AnnualProfile />}
      </div>
    </div>
  )
}
