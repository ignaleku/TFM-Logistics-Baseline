import { useEffect, useMemo, useState } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer,
} from 'recharts'
import { api } from '../../api'
import type { BottlenecksResponse, FullResult, MonthBottleneckReport, PolicyComparisonEntry } from '../../types'
import { PolicyBadge } from '../PolicyBadge'
import { fmtEuro, fmtPct, fmtDelta } from '../../utils/format'

interface Props {
  results: FullResult[]
}

const POLICIES = ['fifo', 'urgent_first', 'rl3_dqn'] as const
const POLICY_LABELS: Record<string, string> = { fifo: 'FIFO', urgent_first: 'Urgent-First', rl3_dqn: 'RL-3 DQN' }
const POLICY_COLORS: Record<string, string> = { fifo: '#94a3b8', urgent_first: '#f97316', rl3_dqn: '#7c3aed' }

function avg(arr: number[]) {
  return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0
}

function violationText(entry: PolicyComparisonEntry, targets: { urgent_target: number; normal_target: number }): string {
  const parts: string[] = []
  if (entry.urgent_sla < targets.urgent_target) {
    parts.push(`Urgent SLA below target (${fmtPct(entry.urgent_sla)} < ${fmtPct(targets.urgent_target, 0)})`)
  }
  if (entry.normal_sla < targets.normal_target) {
    parts.push(`Normal SLA below target (${fmtPct(entry.normal_sla)} < ${fmtPct(targets.normal_target, 0)})`)
  }
  return parts.join('; ')
}

function PolicyCard({
  entry, targets, isRecommended, cheapestCost,
}: {
  entry: PolicyComparisonEntry
  targets: { urgent_target: number; normal_target: number }
  isRecommended: boolean
  cheapestCost: number
}) {
  const diff = entry.total_cost - cheapestCost
  return (
    <div className={`card relative ${isRecommended ? 'ring-2 ring-indigo-500' : ''}`}>
      {isRecommended && (
        <span className="absolute -top-2.5 left-4 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-indigo-600 text-white">
          Recommended
        </span>
      )}
      <div className="flex items-center justify-between mb-4 mt-1">
        <PolicyBadge policy={entry.policy} />
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
          entry.feasible ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
        }`}>
          {entry.feasible ? 'Feasible' : 'Infeasible'}
        </span>
      </div>

      <div>
        <p className="stat-label">Total Cost</p>
        <p className="text-xl font-bold text-slate-800">{fmtEuro(entry.total_cost)}</p>
        {!isRecommended && Math.abs(diff) > 0.5 && (
          <p className={`text-xs mt-0.5 ${diff > 0 ? 'text-red-500' : 'text-emerald-600'}`}>
            {fmtDelta(diff)} vs recommended
          </p>
        )}
      </div>

      <div className="grid grid-cols-3 gap-2 pt-3 mt-3 border-t border-slate-100">
        <div>
          <p className="stat-label">Total SLA</p>
          <p className="text-base font-semibold">{fmtPct(entry.total_sla)}</p>
        </div>
        <div>
          <p className="stat-label">Urgent SLA</p>
          <p className="text-base font-semibold text-orange-600">{fmtPct(entry.urgent_sla)}</p>
        </div>
        <div>
          <p className="stat-label">Normal SLA</p>
          <p className={`text-base font-semibold ${entry.starvation_pattern ? 'text-red-600' : 'text-sky-600'}`}>{fmtPct(entry.normal_sla)}</p>
        </div>
      </div>

      <p className="text-xs text-slate-400 mt-3">
        {Math.round(entry.late_orders).toLocaleString()} late order(s) — {Math.round(entry.urgent_late_orders)} urgent, {Math.round(entry.normal_late_orders)} normal
      </p>

      {!entry.feasible && (
        <p className="text-xs text-red-600 mt-2 pt-2 border-t border-red-100">
          {violationText(entry, targets)}.
        </p>
      )}
      {entry.starvation_pattern && (
        <p className="text-xs text-amber-700 mt-2 pt-2 border-t border-amber-100 font-medium">
          ⚠ {POLICY_LABELS[entry.policy] ?? entry.policy} strongly prioritises urgent orders and violates the configured normal-SLA floor.
        </p>
      )}
    </div>
  )
}

function SecondarySection({ results, diagnostics }: {
  results: FullResult[]
  diagnostics: MonthBottleneckReport['capacity_level_diagnostics']
}) {
  const stats = useMemo(() => {
    const out: Record<string, { avgCost: number; wins: number }> = {}
    POLICIES.forEach((p) => {
      const rows = results.filter((r) => r.policy === p)
      out[p] = { avgCost: avg(rows.map((r) => r.estimated_total_cost)), wins: 0 }
    })
    const groups = new Map<string, FullResult[]>()
    results.forEach((r) => {
      const k = `${r.month_name}|${r.regime}`
      if (!groups.has(k)) groups.set(k, [])
      groups.get(k)!.push(r)
    })
    groups.forEach((grp) => {
      const best = grp.reduce((a, b) => (a.estimated_total_cost < b.estimated_total_cost ? a : b))
      if (out[best.policy]) out[best.policy].wins++
    })
    return out
  }, [results])

  const barData = POLICIES.map((p) => ({
    policy: POLICY_LABELS[p],
    avgCost: stats[p]?.avgCost ?? 0,
    wins: stats[p]?.wins ?? 0,
  }))

  const radarData = useMemo(() => {
    const maxCost = Math.max(...POLICIES.map((p) => stats[p]?.avgCost ?? 0))
    return [
      { metric: 'Cost Efficiency', ...Object.fromEntries(POLICIES.map((p) => [p, maxCost > 0 ? (1 - (stats[p]?.avgCost ?? 0) / maxCost) * 100 : 50])) },
      { metric: 'Wins', ...Object.fromEntries(POLICIES.map((p) => [p, stats[p]?.wins ?? 0])) },
    ]
  }, [stats])

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-semibold text-slate-600">Performance Across Tested Capacity Levels</p>
        <p className="text-xs text-slate-400 mt-1">
          Diagnostic only — how each policy performs across every tested workforce configuration, not just the
          recommended one. Never used to choose the recommendation above.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {POLICIES.map((p) => {
          const d = diagnostics?.feasible_by_policy?.[p]
          return (
            <div key={p} className="card">
              <div className="flex items-center justify-between mb-2">
                <PolicyBadge policy={p} size="sm" />
                <span className="text-xs text-slate-400">{d ? `${d.feasible_count} / ${d.tested_count} feasible` : '—'}</span>
              </div>
              <p className="text-xs text-slate-400">tested configurations</p>
            </div>
          )
        })}
      </div>

      {diagnostics && (
        <p className="text-xs text-slate-400">
          {diagnostics.base_regimes_tested} base workforce configuration(s) tested
          {diagnostics.adaptive_candidates_tested > 0 && (
            <> · {diagnostics.adaptive_candidates_tested} adaptive candidate(s) tested, {diagnostics.adaptive_candidates_accepted} accepted</>
          )}.
        </p>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Average Total Cost by Policy (all tested regimes)</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="policy" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => fmtEuro(v)} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Bar dataKey="avgCost" name="Avg Total Cost" radius={[4, 4, 4, 4]}>
                {barData.map((_, i) => <Cell key={i} fill={Object.values(POLICY_COLORS)[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Wins by Policy (lowest cost per month×regime)</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={barData} layout="vertical" margin={{ top: 4, right: 32, bottom: 4, left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
              <YAxis dataKey="policy" type="category" tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="wins" name="Wins" radius={[0, 4, 4, 0]}>
                {barData.map((_, i) => <Cell key={i} fill={Object.values(POLICY_COLORS)[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card xl:col-span-2">
          <p className="text-sm font-semibold text-slate-600 mb-4">Multi-metric Radar (all tested regimes)</p>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis tick={{ fontSize: 9 }} />
              {POLICIES.map((p) => (
                <Radar key={p} name={POLICY_LABELS[p]} dataKey={p} stroke={POLICY_COLORS[p]} fill={POLICY_COLORS[p]} fillOpacity={0.15} />
              ))}
              <Legend />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

export function PolicyComparisonTab({ results }: Props) {
  const [data, setData] = useState<BottlenecksResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedMonth, setSelectedMonth] = useState<string>('')

  useEffect(() => {
    api.getLatestBottlenecks()
      .then((res) => {
        setData(res)
        if (res.months.length) setSelectedMonth(res.months[0].month_name)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [])

  const report: MonthBottleneckReport | undefined = useMemo(
    () => data?.months.find((m) => m.month_name === selectedMonth),
    [data, selectedMonth]
  )

  if (loading) return <div className="text-center py-24 text-slate-400">Loading policy comparison…</div>
  if (error) return (
    <div className="p-4 bg-red-50 rounded-xl border border-red-200 text-sm text-red-700">
      <strong>Could not load policy comparison.</strong> {error}
      <p className="mt-1 text-xs">Run a simulation from the Run tab first.</p>
    </div>
  )
  if (!data || !data.months.length || !report) {
    return <div className="text-center py-24 text-slate-400">No results. Run a simulation first.</div>
  }

  const rec = report.selected_recommendation
  const comparison = report.policy_comparison ?? []
  const cheapestCost = Math.min(...comparison.map((c) => c.total_cost))

  const barData = comparison.map((c) => ({
    policy: POLICY_LABELS[c.policy] ?? c.policy,
    totalCost: c.total_cost,
    totalSla: c.total_sla * 100,
    urgentSla: c.urgent_sla * 100,
    normalSla: c.normal_sla * 100,
  }))

  return (
    <div className="space-y-8">
      {/* Month selector */}
      {data.months.length > 1 && (
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-sm font-medium text-slate-500">Month:</span>
          {data.months.map((m) => (
            <button
              key={m.month_name}
              onClick={() => setSelectedMonth(m.month_name)}
              className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
                m.month_name === selectedMonth
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'bg-white border border-slate-200 text-slate-600 hover:border-indigo-300'
              }`}
            >
              {m.month_name}
            </button>
          ))}
        </div>
      )}

      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-slate-800">Policy Comparison at Recommended Workforce</h2>
        <p className="text-sm text-slate-500 mt-1">
          Comparing FIFO, Urgent-First and RL-3 under the same final workforce configuration and simulation
          conditions — identical scenario seeds, arrival times, and per-order service times (common random numbers).
        </p>
      </div>

      {/* Recommended workforce banner */}
      <div className="card border-l-4 border-l-indigo-500">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Recommended workforce:</span>
            <span className="font-mono text-sm font-bold text-slate-700">{rec.regime}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              rec.regime_source === 'adaptive' ? 'bg-sky-100 text-sky-700' : 'bg-slate-100 text-slate-600'
            }`}>
              {rec.regime_source === 'adaptive' ? 'Adaptive capacity' : 'Base regime'}
            </span>
            {!rec.feasible && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-medium">
                Best available — SLA targets not fully met
              </span>
            )}
          </div>
          <div className="flex gap-4 text-xs text-slate-500">
            <span>Picking: <strong className="text-slate-700">{rec.picking_workers}</strong></span>
            <span>Packing: <strong className="text-slate-700">{rec.packing_workers}</strong></span>
            <span>Dispatch: <strong className="text-slate-700">{rec.dispatch_workers}</strong></span>
          </div>
        </div>
      </div>

      {/* Primary policy cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {comparison.map((c) => (
          <PolicyCard
            key={c.policy}
            entry={c}
            targets={report.sla_targets}
            isRecommended={c.policy === report.recommended_policy}
            cheapestCost={cheapestCost}
          />
        ))}
      </div>

      {/* Primary charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Total Cost by Policy</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="policy" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => fmtEuro(v)} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Bar dataKey="totalCost" name="Total Cost" radius={[4, 4, 4, 4]}>
                {barData.map((_, i) => <Cell key={i} fill={Object.values(POLICY_COLORS)[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">SLA by Policy</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="policy" tick={{ fontSize: 11 }} />
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

      {/* Secondary: performance across tested capacity levels */}
      <div className="pt-2 border-t border-slate-100">
        <SecondarySection results={results} diagnostics={report.capacity_level_diagnostics} />
      </div>
    </div>
  )
}
