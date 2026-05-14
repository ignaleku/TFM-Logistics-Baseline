import { parseRegime, STAGE_LABELS, STAGE_COLORS } from '../utils/regime'

interface Props {
  regime: string
}

const STAGES = ['picking', 'packing', 'dispatch'] as const

export function RegimeChips({ regime }: Props) {
  const parsed = parseRegime(regime)
  if (!parsed) return <span className="text-xs text-slate-400">{regime}</span>

  return (
    <div className="flex flex-wrap gap-1.5">
      {STAGES.map((s) => (
        <span
          key={s}
          className={`inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-lg ${STAGE_COLORS[s]}`}
        >
          <span className="font-bold">{parsed[s]}</span>
          <span>{STAGE_LABELS[s]}</span>
        </span>
      ))}
    </div>
  )
}
