import { PolicyBadge } from './PolicyBadge'
import { RegimeChips } from './RegimeChips'
import { fmtEuro, fmtPct, fmtNum, fmtDelta } from '../utils/format'

export interface CardData {
  title: string
  accent: string
  regime: string
  policy: string
  workers: number
  totalSla: number | null
  urgentSla: number | null
  normalSla: number | null
  lateCost: number | null
  labourCost: number | null
  totalCost: number | null
  gapVsCheapest?: number | null
  tag?: string
}

const ACCENT: Record<string, string> = {
  indigo: 'border-l-indigo-500',
  violet: 'border-l-violet-500',
  green:  'border-l-emerald-500',
  orange: 'border-l-orange-500',
  sky:    'border-l-sky-500',
}

export function RecommendationCard({ data }: { data: CardData }) {
  const border = ACCENT[data.accent] ?? 'border-l-slate-400'

  return (
    <div className={`card border-l-4 ${border} flex flex-col gap-4`}>
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">{data.title}</p>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono text-sm font-bold text-slate-700">{data.regime}</span>
            <PolicyBadge policy={data.policy} size="sm" />
            {data.tag && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-500">{data.tag}</span>
            )}
          </div>
        </div>
        <div className="text-right">
          <p className="stat-label">Total Workers</p>
          <p className="text-3xl font-black text-slate-800">{fmtNum(data.workers)}</p>
        </div>
      </div>

      {/* Regime breakdown */}
      <RegimeChips regime={data.regime} />

      {/* SLA row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-slate-50 rounded-xl p-3 text-center">
          <p className="stat-label">Total SLA</p>
          <p className="text-xl font-bold text-slate-800">{fmtPct(data.totalSla)}</p>
        </div>
        <div className="bg-orange-50 rounded-xl p-3 text-center">
          <p className="stat-label">Urgent SLA</p>
          <p className="text-xl font-bold text-orange-700">{fmtPct(data.urgentSla)}</p>
        </div>
        <div className="bg-sky-50 rounded-xl p-3 text-center">
          <p className="stat-label">Normal SLA</p>
          <p className="text-xl font-bold text-sky-700">{fmtPct(data.normalSla)}</p>
        </div>
      </div>

      {/* Cost row */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <p className="stat-label">SLA Penalty Cost</p>
          <p className="text-base font-semibold text-red-600">{fmtEuro(data.lateCost)}</p>
        </div>
        <div>
          <p className="stat-label">Labour Cost</p>
          <p className="text-base font-semibold text-slate-700">{fmtEuro(data.labourCost)}</p>
        </div>
        <div>
          <p className="stat-label">Total Estimated Cost</p>
          <p className="text-base font-bold text-slate-900">{fmtEuro(data.totalCost)}</p>
        </div>
      </div>

      {/* Gap */}
      {data.gapVsCheapest != null && (
        <div className="flex items-center gap-2 pt-1 border-t border-slate-100">
          <span className="text-xs text-slate-400">vs Cheapest:</span>
          <span className={`text-sm font-semibold ${data.gapVsCheapest > 0 ? 'text-red-500' : 'text-emerald-500'}`}>
            {fmtDelta(data.gapVsCheapest)}
          </span>
        </div>
      )}
    </div>
  )
}
