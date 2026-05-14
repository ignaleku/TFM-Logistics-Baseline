import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer,
} from 'recharts'
import type { MonthSummary } from '../../types'
import { RecommendationCard } from '../RecommendationCard'
import type { CardData } from '../RecommendationCard'
import { fmtEuro } from '../../utils/format'

interface Props {
  summaries: MonthSummary[]
}

const CURRENCY_TICK = (v: number) => fmtEuro(v)

export function WorkforcePlannerTab({ summaries }: Props) {
  const months = useMemo(() => summaries.map((s) => s.month_name), [summaries])
  const [selectedMonth, setSelectedMonth] = useState<string>(months[0] ?? '')

  const selected = useMemo(
    () => summaries.find((s) => s.month_name === selectedMonth),
    [summaries, selectedMonth]
  )

  const cards = useMemo((): CardData[] => {
    if (!selected) return []
    const cards: CardData[] = []

    // 1. Cheapest
    cards.push({
      title: 'Cheapest Option',
      accent: 'indigo',
      regime: selected.best_total_regime,
      policy: selected.best_total_policy,
      workers: selected.best_total_workers,
      totalSla: selected.best_total_sla,
      urgentSla: selected.best_total_urgent_sla ?? null,
      normalSla: selected.best_total_normal_sla ?? null,
      lateCost: selected.best_total_late_cost ?? null,
      labourCost: selected.best_total_labour_cost ?? null,
      totalCost: selected.best_total_cost,
    })

    // 2. Best RL-3
    if (selected.best_rl3_regime) {
      cards.push({
        title: 'Best RL-3 Option',
        accent: 'violet',
        regime: selected.best_rl3_regime,
        policy: 'rl3_dqn',
        workers: selected.best_rl3_workers ?? 0,
        totalSla: selected.best_rl3_sla ?? null,
        urgentSla: selected.best_rl3_urgent_sla ?? null,
        normalSla: selected.best_rl3_normal_sla ?? null,
        lateCost: selected.best_rl3_late_cost ?? null,
        labourCost: selected.best_rl3_labour_cost ?? null,
        totalCost: selected.best_rl3_total_cost ?? null,
        gapVsCheapest: selected.best_rl3_gap_vs_cheapest ?? null,
      })
    }

    // 3. Minimum workforce for urgent SLA >= 95%
    if (selected.min_urgent_regime) {
      cards.push({
        title: 'Minimum Workforce for Urgent SLA ≥ 95%',
        accent: 'green',
        regime: selected.min_urgent_regime,
        policy: selected.min_urgent_policy ?? '',
        workers: selected.min_urgent_workers ?? 0,
        totalSla: selected.min_urgent_sla ?? null,
        urgentSla: selected.min_urgent_urgent_sla ?? null,
        normalSla: selected.min_urgent_normal_sla ?? null,
        lateCost: selected.min_urgent_late_cost ?? null,
        labourCost: selected.min_urgent_labour_cost ?? null,
        totalCost: selected.min_urgent_total_cost ?? null,
        tag: 'urgent SLA ≥ 95%',
      })
    }

    // 4. Minimum workforce for total SLA >= 80%
    if (selected.min_total_sla_regime) {
      cards.push({
        title: 'Minimum Workforce for Total SLA ≥ 80%',
        accent: 'sky',
        regime: selected.min_total_sla_regime,
        policy: selected.min_total_sla_policy ?? '',
        workers: selected.min_total_sla_workers ?? 0,
        totalSla: selected.min_total_sla ?? null,
        urgentSla: selected.min_total_sla_urgent_sla ?? null,
        normalSla: selected.min_total_sla_normal_sla ?? null,
        lateCost: selected.min_total_sla_late_cost ?? null,
        labourCost: selected.min_total_sla_labour_cost ?? null,
        totalCost: selected.min_total_sla_total_cost ?? null,
        tag: 'total SLA ≥ 80%',
      })
    }

    return cards
  }, [selected])

  // Chart data
  const costChartData = useMemo(() => {
    if (!selected) return []
    const data = []
    data.push({
      name: 'Cheapest',
      lateCost: selected.best_total_late_cost ?? 0,
      labourCost: selected.best_total_labour_cost ?? selected.best_total_cost ?? 0,
      totalCost: selected.best_total_cost ?? 0,
    })
    if (selected.best_rl3_regime) {
      data.push({
        name: 'Best RL-3',
        lateCost: selected.best_rl3_late_cost ?? 0,
        labourCost: selected.best_rl3_labour_cost ?? 0,
        totalCost: selected.best_rl3_total_cost ?? 0,
      })
    }
    if (selected.min_urgent_regime) {
      data.push({
        name: 'Min Urgent',
        lateCost: selected.min_urgent_late_cost ?? 0,
        labourCost: selected.min_urgent_labour_cost ?? 0,
        totalCost: selected.min_urgent_total_cost ?? 0,
      })
    }
    if (selected.min_total_sla_regime) {
      data.push({
        name: 'Min Total SLA',
        lateCost: selected.min_total_sla_late_cost ?? 0,
        labourCost: selected.min_total_sla_labour_cost ?? 0,
        totalCost: selected.min_total_sla_total_cost ?? 0,
      })
    }
    return data
  }, [selected])

  const trendData = useMemo(() =>
    summaries.map((s) => ({
      month: s.month_name?.slice(0, 3),
      workers: s.best_total_workers,
      totalCost: s.best_total_cost,
    })),
    [summaries]
  )

  if (!summaries.length) {
    return (
      <div className="text-center py-24 text-slate-400">
        No results available. Run a simulation first.
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Month selector */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm font-medium text-slate-500">Month:</span>
        {months.map((m) => (
          <button
            key={m}
            onClick={() => setSelectedMonth(m)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
              m === selectedMonth
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'bg-white border border-slate-200 text-slate-600 hover:border-indigo-300'
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      {/* Interpretation message */}
      {selected?.managerial_interpretation_short && (
        <div className="p-4 bg-indigo-50 rounded-xl border border-indigo-100 text-sm text-indigo-800">
          💡 {selected.managerial_interpretation_short}
        </div>
      )}

      {/* Recommendation cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-2 gap-5">
        {cards.map((c) => (
          <RecommendationCard key={c.title} data={c} />
        ))}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">SLA Penalty vs Labour Cost by Option</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={costChartData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={CURRENCY_TICK} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Legend />
              <Bar dataKey="lateCost" name="SLA Penalty Cost" fill="#f87171" stackId="a" radius={[0, 0, 4, 4]} />
              <Bar dataKey="labourCost" name="Labour Cost" fill="#818cf8" stackId="a" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Total Estimated Cost by Option</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={costChartData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={CURRENCY_TICK} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Bar dataKey="totalCost" name="Total Cost" fill="#4f46e5" radius={[4, 4, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Recommended Workers by Month</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={trendData} margin={{ top: 4, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="workers" name="Workers" fill="#6366f1" radius={[4, 4, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Best Total Cost by Month</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trendData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={CURRENCY_TICK} tick={{ fontSize: 10 }} width={72} />
              <Tooltip formatter={(v: number) => fmtEuro(v)} />
              <Line
                type="monotone"
                dataKey="totalCost"
                name="Total Cost"
                stroke="#4f46e5"
                strokeWidth={2.5}
                dot={{ r: 4, fill: '#4f46e5' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
