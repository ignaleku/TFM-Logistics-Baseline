import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ScatterChart, Scatter, ResponsiveContainer,
} from 'recharts'
import type { FullResult } from '../../types'
import { PolicyBadge } from '../PolicyBadge'
import { fmtEuro, fmtPct } from '../../utils/format'

interface Props {
  results: FullResult[]
}

const POLICY_COLORS: Record<string, string> = {
  fifo: '#94a3b8',
  urgent_first: '#f97316',
  rl3_dqn: '#7c3aed',
}

export function MonthlyResultsTab({ results }: Props) {
  const months = useMemo(() => [...new Set(results.map((r) => r.month_name))].sort(), [results])
  const policies = useMemo(() => [...new Set(results.map((r) => r.policy))], [results])
  const regimes = useMemo(() => [...new Set(results.map((r) => r.regime))].sort(), [results])

  const [filterMonth, setFilterMonth] = useState('all')
  const [filterPolicy, setFilterPolicy] = useState('all')
  const [filterRegime, setFilterRegime] = useState('all')

  const filtered = useMemo(() =>
    results.filter((r) =>
      (filterMonth === 'all' || r.month_name === filterMonth) &&
      (filterPolicy === 'all' || r.policy === filterPolicy) &&
      (filterRegime === 'all' || r.regime === filterRegime)
    ),
    [results, filterMonth, filterPolicy, filterRegime]
  )

  // Chart: total cost by month + policy (best per month/policy)
  const costByMonthData = useMemo(() => {
    const grouped: Record<string, Record<string, number>> = {}
    results.forEach((r) => {
      if (!grouped[r.month_name]) grouped[r.month_name] = {}
      const cur = grouped[r.month_name][r.policy]
      if (cur === undefined || r.estimated_total_cost < cur) {
        grouped[r.month_name][r.policy] = r.estimated_total_cost
      }
    })
    return months.map((m) => ({ month: m.slice(0, 3), ...grouped[m] }))
  }, [results, months])

  // Chart: SLA by month + policy (best SLA per month/policy)
  const slaByMonthData = useMemo(() => {
    const grouped: Record<string, Record<string, number>> = {}
    results.forEach((r) => {
      if (!grouped[r.month_name]) grouped[r.month_name] = {}
      const cur = grouped[r.month_name][r.policy]
      if (cur === undefined || r.total_sla > cur) {
        grouped[r.month_name][r.policy] = r.total_sla
      }
    })
    return months.map((m) => ({ month: m.slice(0, 3), ...grouped[m] }))
  }, [results, months])

  if (!results.length) {
    return <div className="text-center py-24 text-slate-400">No results. Run a simulation first.</div>
  }

  return (
    <div className="space-y-8">
      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        {[
          { label: 'Month', value: filterMonth, setter: setFilterMonth, options: months },
          { label: 'Policy', value: filterPolicy, setter: setFilterPolicy, options: policies },
          { label: 'Regime', value: filterRegime, setter: setFilterRegime, options: regimes },
        ].map(({ label, value, setter, options }) => (
          <div key={label} className="flex items-center gap-2">
            <span className="text-sm text-slate-500">{label}:</span>
            <select
              value={value}
              onChange={(e) => setter(e.target.value)}
              className="input-field w-auto py-1.5 text-sm"
            >
              <option value="all">All</option>
              {options.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </div>
        ))}
        <span className="text-xs text-slate-400 self-center">{filtered.length.toLocaleString()} rows</span>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Best Total Cost by Month & Policy</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={costByMonthData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => fmtEuro(v)} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Legend />
              {policies.map((p) => (
                <Bar key={p} dataKey={p} name={p} fill={POLICY_COLORS[p] ?? '#64748b'} radius={[3, 3, 3, 3]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Best SLA by Month & Policy</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={slaByMonthData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => fmtPct(v)} />
              <Legend />
              {policies.map((p) => (
                <Bar key={p} dataKey={p} name={p} fill={POLICY_COLORS[p] ?? '#64748b'} radius={[3, 3, 3, 3]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Workers vs Total Cost</p>
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="total_workers" name="Workers" type="number" tick={{ fontSize: 11 }} label={{ value: 'Workers', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis dataKey="estimated_total_cost" name="Total Cost" tickFormatter={(v) => fmtEuro(v)} tick={{ fontSize: 10 }} width={72} />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                formatter={(v: number, n: string) => [n === 'estimated_total_cost' ? fmtEuro(v) : v, n]}
              />
              <Legend />
              {policies.map((p) => (
                <Scatter
                  key={p}
                  name={p}
                  data={filtered.filter((r) => r.policy === p).map((r) => ({
                    total_workers: r.total_workers,
                    estimated_total_cost: r.estimated_total_cost,
                  }))}
                  fill={POLICY_COLORS[p] ?? '#64748b'}
                  opacity={0.7}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Late Cost vs Labour Cost</p>
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="estimated_worker_cost" name="Labour Cost" tickFormatter={(v) => fmtEuro(v)} tick={{ fontSize: 10 }} />
              <YAxis dataKey="estimated_late_cost" name="SLA Penalty" tickFormatter={(v) => fmtEuro(v)} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Legend />
              {policies.map((p) => (
                <Scatter
                  key={p}
                  name={p}
                  data={filtered.filter((r) => r.policy === p).map((r) => ({
                    estimated_worker_cost: r.estimated_worker_cost,
                    estimated_late_cost: r.estimated_late_cost,
                  }))}
                  fill={POLICY_COLORS[p] ?? '#64748b'}
                  opacity={0.7}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Table */}
      <div className="card overflow-x-auto">
        <p className="text-sm font-semibold text-slate-600 mb-4">
          Detailed Results ({filtered.length.toLocaleString()} rows)
        </p>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-100">
              {['Month', 'Regime', 'Policy', 'Workers', 'Total SLA', 'Urgent SLA', 'Normal SLA',
                'Late Cost', 'Labour Cost', 'Total Cost'].map((h) => (
                <th key={h} className="text-left py-2 pr-3 font-medium text-slate-400 uppercase tracking-wide whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 500).map((r, i) => (
              <tr key={i} className="border-b border-slate-50 hover:bg-slate-50/50">
                <td className="py-2 pr-3 font-medium">{r.month_name}</td>
                <td className="py-2 pr-3 font-mono">{r.regime}</td>
                <td className="py-2 pr-3"><PolicyBadge policy={r.policy} size="sm" /></td>
                <td className="py-2 pr-3">{r.total_workers}</td>
                <td className="py-2 pr-3">{fmtPct(r.total_sla)}</td>
                <td className="py-2 pr-3">{fmtPct(r.urgent_sla)}</td>
                <td className="py-2 pr-3">{fmtPct(r.normal_sla)}</td>
                <td className="py-2 pr-3">{fmtEuro(r.estimated_late_cost)}</td>
                <td className="py-2 pr-3">{fmtEuro(r.estimated_worker_cost)}</td>
                <td className="py-2">{fmtEuro(r.estimated_total_cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length > 500 && (
          <p className="text-xs text-slate-400 mt-3">Showing first 500 rows of {filtered.length.toLocaleString()}. Use filters to narrow down.</p>
        )}
      </div>
    </div>
  )
}
