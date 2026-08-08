import { useEffect, useMemo, useState } from 'react'
import { api, type Mode } from '../../api'
import type { BottlenecksResponse, MonthBottleneckReport, StageBottleneck } from '../../types'
import { fmtEuro, fmtPct } from '../../utils/format'
import { PolicyBadge } from '../PolicyBadge'
import { ResultsEmptyState } from './ResultsEmptyState'

function severity(score: number): { label: string; cls: string; barCls: string } {
  if (score >= 0.7) return { label: 'Critical', cls: 'text-red-700 bg-red-100', barCls: 'bg-red-500' }
  if (score >= 0.45) return { label: 'High', cls: 'text-orange-700 bg-orange-100', barCls: 'bg-orange-500' }
  if (score >= 0.2) return { label: 'Moderate', cls: 'text-amber-700 bg-amber-100', barCls: 'bg-amber-400' }
  return { label: 'Low', cls: 'text-emerald-700 bg-emerald-100', barCls: 'bg-emerald-400' }
}

function StageCard({ stage }: { stage: StageBottleneck }) {
  const sev = severity(stage.pressure_score)
  return (
    <div className={`card border-l-4 ${stage.is_primary_bottleneck ? 'border-l-red-500' : 'border-l-slate-200'}`}>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm font-semibold text-slate-700">{stage.stage_label}</p>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${sev.cls}`}>{sev.label}</span>
      </div>

      <div className="space-y-2.5">
        <div>
          <div className="flex justify-between text-xs text-slate-500 mb-1">
            <span>Pressure score</span>
            <span className="font-mono">{stage.pressure_score.toFixed(2)}</span>
          </div>
          <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
            <div className={`h-2 rounded-full ${sev.barCls}`} style={{ width: `${Math.min(100, stage.pressure_score * 100)}%` }} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs pt-1">
          <div><span className="text-slate-400">Utilisation:</span> <strong>{fmtPct(stage.utilisation)}</strong></div>
          <div><span className="text-slate-400">Avg wait:</span> <strong>{stage.avg_wait_min.toFixed(1)}m</strong></div>
          <div><span className="text-slate-400">p95 wait:</span> <strong>{stage.p95_wait_min.toFixed(1)}m</strong></div>
          <div><span className="text-slate-400">Avg queue:</span> <strong>{stage.avg_queue_len.toFixed(1)}</strong></div>
          <div className="col-span-2"><span className="text-slate-400">Late-order wait share:</span> <strong>{fmtPct(stage.late_wait_share)}</strong></div>
        </div>
      </div>
    </div>
  )
}

interface Props {
  mode: Mode
  onGoToRun: () => void
}

export function CapacityBottlenecksContent({ mode, onGoToRun }: Props) {
  const [data, setData] = useState<BottlenecksResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedMonth, setSelectedMonth] = useState<string>('')

  useEffect(() => {
    setLoading(true)
    setNotFound(false)
    setError(null)
    api.getLatestBottlenecks(mode)
      .then((res) => {
        setData(res)
        if (res.months.length) setSelectedMonth(res.months[0].month_name)
      })
      .catch((e) => {
        const msg = e instanceof Error ? e.message : String(e)
        if (/No .* bottleneck analysis available/i.test(msg)) setNotFound(true)
        else setError(msg)
      })
      .finally(() => setLoading(false))
  }, [mode])

  const report: MonthBottleneckReport | undefined = useMemo(
    () => data?.months.find((m) => m.month_name === selectedMonth),
    [data, selectedMonth]
  )

  if (loading) return <div className="text-center py-24 text-slate-400">Loading bottleneck analysis…</div>
  if (error) return (
    <div className="p-4 bg-red-50 rounded-xl border border-red-200 text-sm text-red-700">
      <strong>Could not load bottleneck analysis.</strong> {error}
    </div>
  )
  if (notFound || !data || !data.months.length || !report) {
    return <ResultsEmptyState label={mode === 'future' ? 'Future Planning' : 'Historical Analysis'} onGoToRun={onGoToRun} />
  }

  const rec = report.selected_recommendation
  const be = report.break_even
  const adaptive = report.adaptive_search

  return (
    <div className="space-y-8">
      {/* Month selector (historical, multi-month) */}
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
      {mode === 'future' && (
        <span className="inline-block text-xs px-2.5 py-1 rounded-full bg-violet-100 text-violet-700 font-medium">
          Future-planning scenario — expectation over {report.replication_count ?? report.scenario_preview?.replications ?? 1} simulated replications
        </span>
      )}

      {/* A. Primary diagnosis */}
      <div className="card border-l-4 border-l-indigo-500">
        <div className="flex items-start justify-between flex-wrap gap-3 mb-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Primary Diagnosis</p>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-sm font-bold text-slate-700">{rec.regime}</span>
              <PolicyBadge policy={rec.policy} size="sm" />
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                rec.feasible ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
              }`}>
                {rec.feasible ? 'Feasible' : 'Best available — SLA targets not fully met'}
              </span>
              {rec.regime_source === 'adaptive' && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-sky-100 text-sky-700 font-medium">from adaptive search</span>
              )}
            </div>
          </div>
          <div className="text-right">
            <p className="stat-label">Total Workers</p>
            <p className="text-2xl font-black text-slate-800">
              {rec.picking_workers + rec.packing_workers + rec.dispatch_workers}
            </p>
          </div>
        </div>
        <p className="text-sm text-slate-600">{report.explanation}</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          <div className="bg-slate-50 rounded-xl p-3 text-center">
            <p className="stat-label">Total SLA</p>
            <p className="text-lg font-bold text-slate-800">{fmtPct(rec.total_sla)}</p>
          </div>
          <div className="bg-orange-50 rounded-xl p-3 text-center">
            <p className="stat-label">Urgent SLA</p>
            <p className="text-lg font-bold text-orange-700">{fmtPct(rec.urgent_sla)}</p>
            <p className="text-[10px] text-slate-400">target {fmtPct(report.sla_targets.urgent_target, 0)}</p>
          </div>
          <div className="bg-sky-50 rounded-xl p-3 text-center">
            <p className="stat-label">Normal SLA</p>
            <p className="text-lg font-bold text-sky-700">{fmtPct(rec.normal_sla)}</p>
            <p className="text-[10px] text-slate-400">target {fmtPct(report.sla_targets.normal_target, 0)}</p>
          </div>
          <div className="bg-slate-50 rounded-xl p-3 text-center">
            <p className="stat-label">Estimated Cost</p>
            <p className="text-lg font-bold text-slate-800">{fmtEuro(rec.estimated_total_cost)}</p>
          </div>
        </div>
      </div>

      {/* B. Stage pressure cards */}
      <div>
        <p className="text-sm font-semibold text-slate-600 mb-3">Stage Pressure</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {report.bottleneck_ranking.map((s) => <StageCard key={s.stage} stage={s} />)}
        </div>
      </div>

      {/* C. Extra-worker economics */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-1">One Extra Worker — Economics</p>
        <p className="text-xs text-slate-400 mb-4">
          +1 worker = +1 monthly FTE at {fmtEuro(be.worker_monthly_cost)}/month (same operating hours used by the
          simulation clock). Theoretical break-even: how many late orders that cost would need to prevent to pay
          for itself — not a guarantee; see the simulated marginal impact below when adaptive search ran.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-slate-50 rounded-xl p-3 text-center">
            <p className="stat-label">Worker Monthly Cost</p>
            <p className="text-base font-bold text-slate-800">{fmtEuro(be.worker_monthly_cost)}</p>
          </div>
          <div className="bg-slate-50 rounded-xl p-3 text-center">
            <p className="stat-label">Urgent-only Break-even</p>
            <p className="text-base font-bold text-slate-800">{be.urgent_only_break_even_orders ?? '—'} orders</p>
          </div>
          <div className="bg-slate-50 rounded-xl p-3 text-center">
            <p className="stat-label">Normal-only Break-even</p>
            <p className="text-base font-bold text-slate-800">{be.normal_only_break_even_orders ?? '—'} orders</p>
          </div>
          <div className="bg-indigo-50 rounded-xl p-3 text-center">
            <p className="stat-label">Mixed Break-even (current mix)</p>
            <p className="text-base font-bold text-indigo-700">{be.mixed_break_even_orders ?? '—'} orders</p>
          </div>
        </div>
      </div>

      {/* D. Adaptive search trail — shown only if triggered */}
      {adaptive.triggered && adaptive.trail && adaptive.trail.length > 0 && (
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-1">Adaptive Capacity Search</p>
          <p className="text-xs text-slate-400 mb-4">
            Stop reason: {adaptive.stop_reason}
            {adaptive.regime_changed && (
              <> — recommendation moved from base search to <strong className="font-mono">{adaptive.final_regime}</strong> ({adaptive.final_policy}).</>
            )}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="text-slate-400 border-b border-slate-100">
                <tr>
                  <th className="py-2 pr-3">#</th>
                  <th className="py-2 pr-3">Parent → Candidate</th>
                  <th className="py-2 pr-3">Added Stage</th>
                  <th className="py-2 pr-3">Policy</th>
                  <th className="py-2 pr-3">Total Cost Δ</th>
                  <th className="py-2 pr-3">Urgent SLA</th>
                  <th className="py-2 pr-3">Normal SLA</th>
                  <th className="py-2 pr-3">Result</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {adaptive.trail.map((t, i) => (
                  <tr key={i} className={t.accepted ? '' : 'opacity-50'}>
                    <td className="py-2 pr-3 text-slate-400">{t.iteration}</td>
                    <td className="py-2 pr-3 font-mono">{t.parent_regime} → {t.candidate_regime}</td>
                    <td className="py-2 pr-3 capitalize">{t.added_stage}</td>
                    <td className="py-2 pr-3"><PolicyBadge policy={t.policy} size="sm" /></td>
                    <td className={`py-2 pr-3 font-semibold ${t.total_cost_diff <= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                      {t.total_cost_diff >= 0 ? '+' : ''}{fmtEuro(t.total_cost_diff)}
                    </td>
                    <td className="py-2 pr-3">{fmtPct(t.urgent_sla_before)} → {fmtPct(t.urgent_sla_after)}</td>
                    <td className="py-2 pr-3">{fmtPct(t.normal_sla_before)} → {fmtPct(t.normal_sla_after)}</td>
                    <td className="py-2 pr-3">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        t.accepted ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500'
                      }`}>
                        {t.accepted ? 'Accepted' : 'Rejected'}
                      </span>
                      <p className="text-[10px] text-slate-400 mt-0.5">{t.reason}</p>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {!adaptive.triggered && (
        <div className="p-3 bg-slate-50 rounded-xl text-xs text-slate-500">
          Adaptive capacity search was not triggered — the recommended configuration already meets SLA targets
          without high pressure and late orders on its bottleneck stage.
        </div>
      )}
    </div>
  )
}
