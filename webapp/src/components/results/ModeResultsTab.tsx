import { useEffect, useState } from 'react'
import { api, type Mode } from '../../api'
import type { RunContext } from '../../types'
import { ContextBanner } from './ContextBanner'
import { RecommendationsContent } from './RecommendationsContent'
import { DemandComplexityContent } from './DemandComplexityContent'
import { PolicyComparisonContent } from './PolicyComparisonContent'
import { CapacityBottlenecksContent } from './CapacityBottlenecksContent'

const SUBTABS = [
  { id: 'recommendation', futureLabel: 'Recommendation', historicalLabel: 'Recommendations' },
  { id: 'demand', futureLabel: 'Demand & Complexity', historicalLabel: 'Demand & Complexity' },
  { id: 'policy', futureLabel: 'Policy Comparison', historicalLabel: 'Policy Comparison' },
  { id: 'capacity', futureLabel: 'Capacity & Bottlenecks', historicalLabel: 'Capacity & Bottlenecks' },
] as const

type SubtabId = typeof SUBTABS[number]['id']

interface Props {
  mode: Mode
  refreshKey: number
  onGoToRun: () => void
}

export function ModeResultsTab({ mode, refreshKey, onGoToRun }: Props) {
  const [subtab, setSubtab] = useState<SubtabId>('recommendation')
  const [ctx, setCtx] = useState<RunContext | null>(null)

  useEffect(() => {
    api.getLatestContext(mode).then(setCtx).catch(() => setCtx(null))
  }, [mode, refreshKey])

  return (
    <div className="space-y-6">
      {ctx && <ContextBanner ctx={ctx} />}

      <nav className="flex gap-1 overflow-x-auto bg-slate-100 rounded-xl p-1 w-fit">
        {SUBTABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setSubtab(t.id)}
            className={`tab-btn ${subtab === t.id ? 'tab-btn-active' : 'tab-btn-inactive'}`}
          >
            {mode === 'future' ? t.futureLabel : t.historicalLabel}
          </button>
        ))}
      </nav>

      {/* key forces each content component to remount (and re-fetch) on mode switch or when a
          new run for this mode completes, rather than relying solely on prop-change effects. */}
      {subtab === 'recommendation' && <RecommendationsContent key={`${mode}-${refreshKey}`} mode={mode} onGoToRun={onGoToRun} />}
      {subtab === 'demand' && <DemandComplexityContent key={`${mode}-${refreshKey}`} mode={mode} onGoToRun={onGoToRun} />}
      {subtab === 'policy' && <PolicyComparisonContent key={`${mode}-${refreshKey}`} mode={mode} onGoToRun={onGoToRun} />}
      {subtab === 'capacity' && <CapacityBottlenecksContent key={`${mode}-${refreshKey}`} mode={mode} onGoToRun={onGoToRun} />}
    </div>
  )
}
