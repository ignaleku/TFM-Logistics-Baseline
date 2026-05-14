import { useState, useMemo } from 'react'
import type { FullResult, MonthSummary } from '../../types'
import { PolicyBadge } from '../PolicyBadge'
import { fmtEuro, fmtPct } from '../../utils/format'

interface Props {
  summaries: MonthSummary[]
  results: FullResult[]
}

export function DataExplorerTab({ summaries, results }: Props) {
  const [activeTable, setActiveTable] = useState<'recommendations' | 'full'>('recommendations')
  const [filterMonth, setFilterMonth] = useState('all')
  const [filterPolicy, setFilterPolicy] = useState('all')
  const [filterRegime, setFilterRegime] = useState('all')
  const [search, setSearch] = useState('')

  const months = useMemo(() => [...new Set([
    ...summaries.map((s) => s.month_name),
    ...results.map((r) => r.month_name),
  ])].sort(), [summaries, results])

  const policies = useMemo(() => [...new Set(results.map((r) => r.policy))], [results])
  const regimes = useMemo(() => [...new Set(results.map((r) => r.regime))].sort(), [results])

  const filteredResults = useMemo(() =>
    results.filter((r) =>
      (filterMonth === 'all' || r.month_name === filterMonth) &&
      (filterPolicy === 'all' || r.policy === filterPolicy) &&
      (filterRegime === 'all' || r.regime === filterRegime)
    ),
    [results, filterMonth, filterPolicy, filterRegime]
  )

  const filteredSummaries = useMemo(() =>
    summaries.filter((s) =>
      filterMonth === 'all' || s.month_name === filterMonth
    ),
    [summaries, filterMonth]
  )

  return (
    <div className="space-y-6">
      {/* Table selector */}
      <div className="flex gap-2">
        {(['recommendations', 'full'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setActiveTable(t)}
            className={`tab-btn ${activeTable === t ? 'tab-btn-active' : 'tab-btn-inactive'}`}
          >
            {t === 'recommendations' ? 'Monthly Recommendations' : 'Full Results'}
          </button>
        ))}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <input
          type="text"
          placeholder="Search…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input-field w-48 py-1.5 text-sm"
        />
        <select
          value={filterMonth}
          onChange={(e) => setFilterMonth(e.target.value)}
          className="input-field w-auto py-1.5 text-sm"
        >
          <option value="all">All months</option>
          {months.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        {activeTable === 'full' && (
          <>
            <select
              value={filterPolicy}
              onChange={(e) => setFilterPolicy(e.target.value)}
              className="input-field w-auto py-1.5 text-sm"
            >
              <option value="all">All policies</option>
              {policies.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
            <select
              value={filterRegime}
              onChange={(e) => setFilterRegime(e.target.value)}
              className="input-field w-auto py-1.5 text-sm"
            >
              <option value="all">All regimes</option>
              {regimes.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </>
        )}
      </div>

      {/* Recommendations table */}
      {activeTable === 'recommendations' && (
        <div className="card overflow-x-auto">
          <p className="text-xs text-slate-400 mb-3">{filteredSummaries.length} months</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-100">
                {['Month', 'Orders', 'Cheapest Regime', 'Policy', 'Workers', 'Total Cost', 'SLA',
                  'RL-3 Regime', 'RL-3 Cost', 'RL-3 Gap'].map((h) => (
                  <th key={h} className="text-left py-2 pr-3 font-medium text-slate-400 uppercase tracking-wide whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredSummaries
                .filter((s) => !search || JSON.stringify(s).toLowerCase().includes(search.toLowerCase()))
                .map((s) => (
                <tr key={s.month} className="border-b border-slate-50 hover:bg-slate-50/50">
                  <td className="py-2 pr-3 font-medium">{s.month_name}</td>
                  <td className="py-2 pr-3">{s.total_orders?.toLocaleString()}</td>
                  <td className="py-2 pr-3 font-mono">{s.best_total_regime}</td>
                  <td className="py-2 pr-3"><PolicyBadge policy={s.best_total_policy} size="sm" /></td>
                  <td className="py-2 pr-3">{s.best_total_workers}</td>
                  <td className="py-2 pr-3">{fmtEuro(s.best_total_cost)}</td>
                  <td className="py-2 pr-3">{fmtPct(s.best_total_sla)}</td>
                  <td className="py-2 pr-3 font-mono">{s.best_rl3_regime}</td>
                  <td className="py-2 pr-3">{fmtEuro(s.best_rl3_total_cost)}</td>
                  <td className="py-2">{s.best_rl3_gap_vs_cheapest != null ? fmtEuro(s.best_rl3_gap_vs_cheapest) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Full results table */}
      {activeTable === 'full' && (
        <div className="card overflow-x-auto">
          <p className="text-xs text-slate-400 mb-3">
            {filteredResults.length.toLocaleString()} rows (showing up to 500)
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-100">
                {['Month', 'Regime', 'Policy', 'Workers', 'Total SLA', 'Urgent SLA',
                  'Late Cost', 'Labour Cost', 'Total Cost'].map((h) => (
                  <th key={h} className="text-left py-2 pr-3 font-medium text-slate-400 uppercase tracking-wide whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredResults
                .filter((r) => !search || JSON.stringify(r).toLowerCase().includes(search.toLowerCase()))
                .slice(0, 500)
                .map((r, i) => (
                <tr key={i} className="border-b border-slate-50 hover:bg-slate-50/50">
                  <td className="py-2 pr-3 font-medium">{r.month_name}</td>
                  <td className="py-2 pr-3 font-mono">{r.regime}</td>
                  <td className="py-2 pr-3"><PolicyBadge policy={r.policy} size="sm" /></td>
                  <td className="py-2 pr-3">{r.total_workers}</td>
                  <td className="py-2 pr-3">{fmtPct(r.total_sla)}</td>
                  <td className="py-2 pr-3">{fmtPct(r.urgent_sla)}</td>
                  <td className="py-2 pr-3">{fmtEuro(r.estimated_late_cost)}</td>
                  <td className="py-2 pr-3">{fmtEuro(r.estimated_worker_cost)}</td>
                  <td className="py-2">{fmtEuro(r.estimated_total_cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
