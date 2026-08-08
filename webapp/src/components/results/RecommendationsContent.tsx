import { useEffect, useMemo, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line, ResponsiveContainer,
} from 'recharts'
import { api, type Mode } from '../../api'
import type { BottlenecksResponse, MonthBottleneckReport, PolicyComparisonEntry } from '../../types'
import { PolicyBadge } from '../PolicyBadge'
import { RegimeChips } from '../RegimeChips'
import { fmtEuro, fmtPct } from '../../utils/format'
import { ResultsEmptyState } from './ResultsEmptyState'

interface Props {
  mode: Mode
  onGoToRun: () => void
}

function bestManualAlternative(comparison: PolicyComparisonEntry[], recommendedPolicy: string): PolicyComparisonEntry | undefined {
  const manual = comparison.filter((c) => c.policy !== 'rl3_dqn' && c.policy !== recommendedPolicy)
  const pool = manual.length ? manual : comparison.filter((c) => c.policy !== recommendedPolicy)
  if (!pool.length) return undefined
  const feasible = pool.filter((c) => c.feasible)
  if (feasible.length) return feasible.reduce((a, b) => (a.total_cost < b.total_cost ? a : b))
  return pool.reduce((a, b) => (a.sla_violation < b.sla_violation ? a : b))
}

function capacityStatusText(report: MonthBottleneckReport): string {
  const rec = report.selected_recommendation
  const ad = report.adaptive_search
  if (ad.triggered && ad.regime_changed) {
    return `Adaptive capacity search expanded the workforce from ${ad.trail?.[0]?.parent_regime ?? 'the initial candidate'} to ${ad.final_regime} to reach a better outcome.`
  }
  if (ad.triggered && !ad.regime_changed) {
    return 'Adaptive capacity search ran but did not find a better configuration — the initial recommendation stands.'
  }
  if (rec.feasible) {
    return 'Minimum feasible workforce reached from the dynamic candidate set. No extra capacity search was needed.'
  }
  return 'Best available configuration from the tested candidates; SLA targets are not fully met.'
}

function SingleMonthRecommendation({ mode, report }: { mode: Mode; report: MonthBottleneckReport }) {
  const rec = report.selected_recommendation
  const comparison = report.policy_comparison ?? []
  const alt = bestManualAlternative(comparison, rec.policy)
  const labourCost = report.break_even.worker_monthly_cost * (rec.picking_workers + rec.packing_workers + rec.dispatch_workers)
  const penaltyCost = Math.max(0, rec.estimated_total_cost - labourCost)
  const isHistorical = mode === 'historical'

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-slate-800">{isHistorical ? 'Recommended Historical Configuration' : 'Recommendation'}</h2>
        <p className="text-sm text-slate-500 mt-1">
          {isHistorical
            ? `What would have performed best for ${report.month_name}, given the observed order mix — a counterfactual estimate, not a guarantee.`
            : `Single-month planning result for ${report.month_name}.`}
        </p>
      </div>

      <div className="card border-l-4 border-l-indigo-500">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <span className="font-mono text-lg font-bold text-slate-800">{rec.regime}</span>
              <PolicyBadge policy={rec.policy} />
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                rec.feasible ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
              }`}>
                {rec.feasible ? 'Feasible' : 'Best available'}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                rec.regime_source === 'adaptive' ? 'bg-sky-100 text-sky-700' : 'bg-slate-100 text-slate-600'
              }`}>
                {rec.regime_source === 'adaptive' ? 'Adaptive capacity' : 'Dynamic candidate'}
              </span>
            </div>
            <RegimeChips regime={rec.regime} />
          </div>
          <div className="text-right">
            <p className="stat-label">Total Workers</p>
            <p className="text-3xl font-black text-slate-800">
              {rec.picking_workers + rec.packing_workers + rec.dispatch_workers}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 mt-5">
          <div className="bg-slate-50 rounded-xl p-3 text-center">
            <p className="stat-label">Total SLA</p>
            <p className="text-xl font-bold text-slate-800">{fmtPct(rec.total_sla)}</p>
          </div>
          <div className="bg-orange-50 rounded-xl p-3 text-center">
            <p className="stat-label">Urgent SLA</p>
            <p className="text-xl font-bold text-orange-700">{fmtPct(rec.urgent_sla)}</p>
            <p className="text-[10px] text-slate-400">target {fmtPct(report.sla_targets.urgent_target, 0)}</p>
          </div>
          <div className="bg-sky-50 rounded-xl p-3 text-center">
            <p className="stat-label">Normal SLA</p>
            <p className="text-xl font-bold text-sky-700">{fmtPct(rec.normal_sla)}</p>
            <p className="text-[10px] text-slate-400">target {fmtPct(report.sla_targets.normal_target, 0)}</p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 mt-3">
          <div>
            <p className="stat-label">Expected Labour Cost</p>
            <p className="text-base font-semibold text-slate-700">{fmtEuro(labourCost)}</p>
          </div>
          <div>
            <p className="stat-label">Expected SLA Penalty</p>
            <p className="text-base font-semibold text-red-600">{fmtEuro(penaltyCost)}</p>
          </div>
          <div>
            <p className="stat-label">Expected Total Cost</p>
            <p className="text-base font-bold text-slate-900">{fmtEuro(rec.estimated_total_cost)}</p>
          </div>
        </div>
      </div>

      {alt && (
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-1">Best Manual Alternative</p>
          <p className="text-xs text-slate-400 mb-4">
            The best FIFO / Urgent-First policy at the same recommended workforce, for comparison against the RL-3 recommendation.
          </p>
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-2">
              <PolicyBadge policy={alt.policy} />
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                alt.feasible ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
              }`}>
                {alt.feasible ? 'Feasible' : 'Infeasible'}
              </span>
            </div>
            <div className="flex gap-5 text-xs text-slate-500">
              <span>Total SLA: <strong className="text-slate-700">{fmtPct(alt.total_sla)}</strong></span>
              <span>Urgent SLA: <strong className="text-slate-700">{fmtPct(alt.urgent_sla)}</strong></span>
              <span>Normal SLA: <strong className="text-slate-700">{fmtPct(alt.normal_sla)}</strong></span>
              <span>Total Cost: <strong className="text-slate-700">{fmtEuro(alt.total_cost)}</strong>{' '}
                <span className={alt.total_cost >= rec.estimated_total_cost ? 'text-red-500' : 'text-emerald-600'}>
                  ({alt.total_cost >= rec.estimated_total_cost ? '+' : ''}{fmtEuro(alt.total_cost - rec.estimated_total_cost)} vs recommended)
                </span>
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-1">Capacity Status</p>
        <p className="text-sm text-slate-600">{capacityStatusText(report)}</p>
      </div>
    </div>
  )
}

function MultiMonthRecommendations({ months }: { months: MonthBottleneckReport[] }) {
  const barData = months.map((m) => ({
    month: m.month_name.slice(0, 3),
    workers: m.selected_recommendation.picking_workers + m.selected_recommendation.packing_workers + m.selected_recommendation.dispatch_workers,
    cost: m.selected_recommendation.estimated_total_cost,
    totalSla: m.selected_recommendation.total_sla * 100,
    urgentSla: m.selected_recommendation.urgent_sla * 100,
    normalSla: m.selected_recommendation.normal_sla * 100,
  }))

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-bold text-slate-800">Recommended Historical Configuration by Month</h2>
        <p className="text-sm text-slate-500 mt-1">
          What would have performed best for each analysed month — counterfactual estimates, not guarantees.
        </p>
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-xs text-left min-w-[760px]">
          <thead className="text-slate-400 border-b border-slate-100">
            <tr>
              <th className="py-2 pr-3">Month</th>
              <th className="py-2 pr-3">Recommended Workforce</th>
              <th className="py-2 pr-3">Policy</th>
              <th className="py-2 pr-3">Total SLA</th>
              <th className="py-2 pr-3">Urgent SLA</th>
              <th className="py-2 pr-3">Normal SLA</th>
              <th className="py-2 pr-3">Cost</th>
              <th className="py-2 pr-3">Primary Bottleneck</th>
              <th className="py-2 pr-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {months.map((m) => {
              const rec = m.selected_recommendation
              return (
                <tr key={m.month_name}>
                  <td className="py-2 pr-3 font-medium text-slate-700">{m.month_name}</td>
                  <td className="py-2 pr-3 font-mono">{rec.regime}</td>
                  <td className="py-2 pr-3"><PolicyBadge policy={rec.policy} size="sm" /></td>
                  <td className="py-2 pr-3">{fmtPct(rec.total_sla)}</td>
                  <td className="py-2 pr-3">{fmtPct(rec.urgent_sla)}</td>
                  <td className="py-2 pr-3">{fmtPct(rec.normal_sla)}</td>
                  <td className="py-2 pr-3">{fmtEuro(rec.estimated_total_cost)}</td>
                  <td className="py-2 pr-3 capitalize">{m.primary_bottleneck}</td>
                  <td className="py-2 pr-3">
                    <span className={`px-2 py-0.5 rounded-full font-medium ${
                      rec.feasible ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
                    }`}>
                      {rec.feasible ? 'Feasible' : 'Best available'}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Recommended Workers by Month</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="workers" name="Workers" fill="#6366f1" radius={[4, 4, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Cost by Month</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => fmtEuro(v)} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Line type="monotone" dataKey="cost" name="Total Cost" stroke="#4f46e5" strokeWidth={2.5} dot={{ r: 4, fill: '#4f46e5' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card xl:col-span-2">
          <p className="text-sm font-semibold text-slate-600 mb-4">SLA by Month</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => `${v.toFixed(0)}%`} domain={[0, 100]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="totalSla" name="Total SLA %" fill="#6366f1" radius={[4, 4, 4, 4]} />
              <Bar dataKey="urgentSla" name="Urgent SLA %" fill="#f97316" radius={[4, 4, 4, 4]} />
              <Bar dataKey="normalSla" name="Normal SLA %" fill="#0ea5e9" radius={[4, 4, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

export function RecommendationsContent({ mode, onGoToRun }: Props) {
  const [data, setData] = useState<BottlenecksResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setNotFound(false)
    setError(null)
    api.getLatestBottlenecks(mode)
      .then(setData)
      .catch((e) => {
        const msg = e instanceof Error ? e.message : String(e)
        if (/No .* bottleneck analysis available/i.test(msg)) setNotFound(true)
        else setError(msg)
      })
      .finally(() => setLoading(false))
  }, [mode])

  const feasibleWarning = useMemo(() => {
    if (!data?.months.length) return null
    if (data.months.length > 1) return null
    const rec = data.months[0].selected_recommendation
    if (rec.feasible) return null
    return `Best available configuration; SLA targets not fully met (urgent target ${fmtPct(data.months[0].sla_targets.urgent_target, 0)}, normal target ${fmtPct(data.months[0].sla_targets.normal_target, 0)}). See Capacity & Bottlenecks for the adaptive capacity search trail.`
  }, [data])

  if (loading) return <div className="text-center py-24 text-slate-400">Loading recommendation…</div>
  if (error) return (
    <div className="p-4 bg-red-50 rounded-xl border border-red-200 text-sm text-red-700">
      <strong>Could not load recommendation.</strong> {error}
    </div>
  )
  if (notFound || !data || !data.months.length) {
    return <ResultsEmptyState label={mode === 'future' ? 'Future Planning' : 'Historical Analysis'} onGoToRun={onGoToRun} />
  }

  return (
    <div className="space-y-6">
      {feasibleWarning && (
        <div className="p-4 bg-amber-50 rounded-xl border border-amber-200 text-sm text-amber-800">
          ⚠ {feasibleWarning}
        </div>
      )}
      {data.months.length <= 1
        ? <SingleMonthRecommendation mode={mode} report={data.months[0]} />
        : <MultiMonthRecommendations months={data.months} />}
    </div>
  )
}

