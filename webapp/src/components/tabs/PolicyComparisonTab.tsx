import { useMemo } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer,
} from 'recharts'
import type { FullResult } from '../../types'
import { PolicyBadge } from '../PolicyBadge'
import { fmtEuro, fmtPct } from '../../utils/format'

interface Props {
  results: FullResult[]
}

const POLICIES = ['fifo', 'urgent_first', 'rl5_dqn'] as const
const POLICY_COLORS = { fifo: '#94a3b8', urgent_first: '#f97316', rl5_dqn: '#7c3aed' }

function avg(arr: number[]) {
  return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0
}

export function PolicyComparisonTab({ results }: Props) {
  const stats = useMemo(() => {
    const out: Record<string, {
      avgCost: number; avgSla: number; avgUrgentSla: number; wins: number
    }> = {}

    POLICIES.forEach((p) => {
      const rows = results.filter((r) => r.policy === p)
      out[p] = {
        avgCost: avg(rows.map((r) => r.estimated_total_cost)),
        avgSla: avg(rows.map((r) => r.total_sla)),
        avgUrgentSla: avg(rows.map((r) => r.urgent_sla)),
        wins: 0,
      }
    })

    // Count wins per (month, regime)
    const groups = new Map<string, FullResult[]>()
    results.forEach((r) => {
      const k = `${r.month_name}|${r.regime}`
      if (!groups.has(k)) groups.set(k, [])
      groups.get(k)!.push(r)
    })
    groups.forEach((grp) => {
      const best = grp.reduce((a, b) => a.estimated_total_cost < b.estimated_total_cost ? a : b)
      if (out[best.policy]) out[best.policy].wins++
    })

    return out
  }, [results])

  const rl5Gap = useMemo(() => {
    if (!stats.rl5_dqn || !stats.urgent_first) return null
    return stats.rl5_dqn.avgCost - stats.urgent_first.avgCost
  }, [stats])

  const barData = useMemo(() =>
    POLICIES.map((p) => ({
      policy: p === 'fifo' ? 'FIFO' : p === 'urgent_first' ? 'Urgent-First' : 'RL-5 DQN',
      avgCost: stats[p]?.avgCost ?? 0,
      avgSla: (stats[p]?.avgSla ?? 0) * 100,
      urgentSla: (stats[p]?.avgUrgentSla ?? 0) * 100,
      wins: stats[p]?.wins ?? 0,
    })),
    [stats]
  )

  const radarData = useMemo(() => {
    const maxCost = Math.max(...POLICIES.map((p) => stats[p]?.avgCost ?? 0))
    return [
      { metric: 'SLA', ...Object.fromEntries(POLICIES.map((p) => [p, (stats[p]?.avgSla ?? 0) * 100])) },
      { metric: 'Urgent SLA', ...Object.fromEntries(POLICIES.map((p) => [p, (stats[p]?.avgUrgentSla ?? 0) * 100])) },
      { metric: 'Cost Efficiency', ...Object.fromEntries(POLICIES.map((p) => [p, maxCost > 0 ? (1 - (stats[p]?.avgCost ?? 0) / maxCost) * 100 : 50])) },
      { metric: 'Wins', ...Object.fromEntries(POLICIES.map((p) => [p, (stats[p]?.wins ?? 0)])) },
    ]
  }, [stats])

  if (!results.length) {
    return <div className="text-center py-24 text-slate-400">No results. Run a simulation first.</div>
  }

  return (
    <div className="space-y-8">
      {/* Disclaimer */}
      <div className="p-4 bg-violet-50 rounded-xl border border-violet-100 text-sm text-violet-800">
        <strong>Note:</strong> RL-5 is a learned dynamic sequencing policy. It is not assumed to always win;
        it is evaluated against manual policies under the same simulation conditions.
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {POLICIES.map((p) => (
          <div key={p} className="card">
            <div className="flex items-center justify-between mb-4">
              <PolicyBadge policy={p} />
              <span className="text-xs text-slate-400 font-medium">{stats[p]?.wins ?? 0} wins</span>
            </div>
            <div className="space-y-2">
              <div>
                <p className="stat-label">Avg Total Cost</p>
                <p className="text-xl font-bold text-slate-800">{fmtEuro(stats[p]?.avgCost)}</p>
              </div>
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-100">
                <div>
                  <p className="stat-label">Avg Total SLA</p>
                  <p className="text-base font-semibold">{fmtPct(stats[p]?.avgSla)}</p>
                </div>
                <div>
                  <p className="stat-label">Avg Urgent SLA</p>
                  <p className="text-base font-semibold text-orange-600">{fmtPct(stats[p]?.avgUrgentSla)}</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* RL-5 gap */}
      {rl5Gap != null && (
        <div className={`p-4 rounded-xl border text-sm font-medium ${
          rl5Gap <= 0
            ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
            : 'bg-amber-50 border-amber-200 text-amber-800'
        }`}>
          RL-5 avg cost vs Urgent-First:{' '}
          <strong>{rl5Gap <= 0 ? 'saves' : 'costs'} {fmtEuro(Math.abs(rl5Gap))}</strong> on average
          {rl5Gap <= 0
            ? ' — RL-5 tends to match or outperform Urgent-First on total cost.'
            : ' — Urgent-First tends to be cheaper; RL-5 may add value in specific months.'}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Average Total Cost by Policy</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="policy" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => fmtEuro(v)} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Bar dataKey="avgCost" name="Avg Total Cost" radius={[4, 4, 4, 4]}>
                {barData.map((_, i) => (
                  <Cell key={i} fill={Object.values(POLICY_COLORS)[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">SLA Comparison by Policy</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="policy" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => `${v.toFixed(0)}%`} domain={[0, 100]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="avgSla" name="Total SLA %" fill="#6366f1" radius={[4, 4, 4, 4]} />
              <Bar dataKey="urgentSla" name="Urgent SLA %" fill="#f97316" radius={[4, 4, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Wins by Policy (lowest cost per month×regime)</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} layout="vertical" margin={{ top: 4, right: 32, bottom: 4, left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
              <YAxis dataKey="policy" type="category" tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="wins" name="Wins" radius={[0, 4, 4, 0]}>
                {barData.map((_, i) => (
                  <Cell key={i} fill={Object.values(POLICY_COLORS)[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Multi-metric Radar</p>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis tick={{ fontSize: 9 }} />
              {POLICIES.map((p) => (
                <Radar
                  key={p}
                  name={p}
                  dataKey={p}
                  stroke={POLICY_COLORS[p]}
                  fill={POLICY_COLORS[p]}
                  fillOpacity={0.15}
                />
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
