import { useMemo } from 'react'
import type { MonthSummary } from '../../types'
import { fmtEuro, fmtNum } from '../../utils/format'
import { PolicyBadge } from '../PolicyBadge'

interface Props {
  summaries: MonthSummary[]
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card flex flex-col gap-1">
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
      {sub && <p className="text-xs text-slate-400">{sub}</p>}
    </div>
  )
}

function mode<T>(arr: T[]): T {
  const freq = new Map<T, number>()
  arr.forEach((v) => freq.set(v, (freq.get(v) ?? 0) + 1))
  return [...freq.entries()].sort((a, b) => b[1] - a[1])[0]?.[0]
}

export function OverviewTab({ summaries }: Props) {
  const stats = useMemo(() => {
    if (!summaries.length) return null
    const totalOrders = summaries.reduce((s, r) => s + (r.total_orders ?? 0), 0)
    const avgWorkers = summaries.reduce((s, r) => s + (r.best_total_workers ?? 0), 0) / summaries.length
    const avgCost = summaries.reduce((s, r) => s + (r.best_total_cost ?? 0), 0) / summaries.length
    const mostCommonPolicy = mode(summaries.map((r) => r.best_total_policy))
    const mostCommonRegime = mode(summaries.map((r) => r.best_total_regime))
    const sorted = [...summaries].sort((a, b) => (b.total_orders ?? 0) - (a.total_orders ?? 0))
    return {
      totalMonths: summaries.length,
      totalOrders,
      avgWorkers: Math.round(avgWorkers),
      avgCost,
      mostCommonPolicy,
      mostCommonRegime,
      highestMonth: sorted[0]?.month_name,
      lowestMonth: sorted[sorted.length - 1]?.month_name,
    }
  }, [summaries])

  if (!summaries.length) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-slate-400">
        <svg className="w-16 h-16 mb-4 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p className="text-lg font-medium">No simulation results yet</p>
        <p className="text-sm mt-1">Upload orders and run a simulation to see the overview.</p>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Stats grid */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Months Analysed" value={fmtNum(stats.totalMonths)} />
          <StatCard label="Total Orders" value={fmtNum(stats.totalOrders)} />
          <StatCard label="Avg Recommended Workers" value={fmtNum(stats.avgWorkers)} />
          <StatCard label="Avg Total Monthly Cost" value={fmtEuro(stats.avgCost)} />
        </div>
      )}

      {/* Key findings */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="card">
            <p className="stat-label mb-3">Most Recommended Policy</p>
            <PolicyBadge policy={stats.mostCommonPolicy} />
          </div>
          <div className="card">
            <p className="stat-label mb-3">Most Recommended Regime</p>
            <span className="font-mono text-lg font-bold text-slate-700">{stats.mostCommonRegime}</span>
          </div>
          <div className="card">
            <p className="stat-label mb-1">Highest Demand Month</p>
            <p className="text-lg font-bold text-slate-800">{stats.highestMonth}</p>
          </div>
          <div className="card">
            <p className="stat-label mb-1">Lowest Demand Month</p>
            <p className="text-lg font-bold text-slate-800">{stats.lowestMonth}</p>
          </div>
        </div>
      )}

      {/* Pipeline diagram */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-6">Optimisation Pipeline</p>
        <div className="flex flex-wrap items-center justify-center gap-3 text-sm">
          {[
            { label: 'Historical Orders CSV', color: 'bg-slate-100 text-slate-700' },
            { label: '+', color: 'text-slate-400 bg-transparent' },
            { label: 'Economic Assumptions', color: 'bg-slate-100 text-slate-700' },
            { label: '→', color: 'text-slate-400 bg-transparent' },
            { label: 'Simulation Engine\n(RL-5 + Baselines)', color: 'bg-indigo-50 text-indigo-700 border border-indigo-200' },
            { label: '→', color: 'text-slate-400 bg-transparent' },
            { label: 'Monthly Recommendation', color: 'bg-emerald-50 text-emerald-700 border border-emerald-200' },
          ].map((s, i) => (
            <div
              key={i}
              className={`px-4 py-2.5 rounded-xl font-medium whitespace-pre-wrap text-center ${s.color}`}
            >
              {s.label}
            </div>
          ))}
        </div>
      </div>

      {/* Monthly summary table */}
      <div className="card overflow-x-auto">
        <p className="text-sm font-semibold text-slate-600 mb-4">Monthly Summary</p>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100">
              {['Month', 'Orders', 'Urgent %', 'Best Policy', 'Regime', 'Workers', 'Total Cost', 'SLA'].map((h) => (
                <th key={h} className="text-left py-2 pr-4 font-medium text-slate-400 text-xs uppercase tracking-wide">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summaries.map((r) => (
              <tr key={r.month} className="border-b border-slate-50 hover:bg-slate-50/50">
                <td className="py-2.5 pr-4 font-medium">{r.month_name}</td>
                <td className="py-2.5 pr-4">{fmtNum(r.total_orders)}</td>
                <td className="py-2.5 pr-4">{r.urgent_share != null ? `${(r.urgent_share * 100).toFixed(1)}%` : '—'}</td>
                <td className="py-2.5 pr-4"><PolicyBadge policy={r.best_total_policy} size="sm" /></td>
                <td className="py-2.5 pr-4 font-mono text-xs">{r.best_total_regime}</td>
                <td className="py-2.5 pr-4 font-bold">{fmtNum(r.best_total_workers)}</td>
                <td className="py-2.5 pr-4">{fmtEuro(r.best_total_cost)}</td>
                <td className="py-2.5">{r.best_total_sla != null ? `${(r.best_total_sla * 100).toFixed(1)}%` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
