export interface RegimeBreakdown {
  picking: number
  packing: number
  dispatch: number
  total: number
}

const STAGES = ['picking', 'packing', 'dispatch'] as const

export function parseRegime(regime: string): RegimeBreakdown | null {
  // Compact format (all single-digit workers): "s321" → {picking:3, packing:2, dispatch:1}
  // Expanded format (any worker count >= 10 in some stage): "s10_6_3" → {picking:10, packing:6, dispatch:3}
  // Mirrors src/analysis/regime_naming.py — the single definition of this format.
  const expanded = regime?.match(/^s(\d+)_(\d+)_(\d+)$/)
  if (expanded) {
    const [, p, pk, d] = expanded.map(Number)
    return { picking: p, packing: pk, dispatch: d, total: p + pk + d }
  }
  const compact = regime?.match(/^s(\d)(\d)(\d)$/)
  if (!compact) return null
  const [, p, pk, d] = compact.map(Number)
  return { picking: p, packing: pk, dispatch: d, total: p + pk + d }
}

export const STAGE_LABELS: Record<typeof STAGES[number], string> = {
  picking:  'Picking',
  packing:  'Packing',
  dispatch: 'Dispatch',
}

export const STAGE_COLORS: Record<typeof STAGES[number], string> = {
  picking:  'bg-indigo-100 text-indigo-700',
  packing:  'bg-sky-100 text-sky-700',
  dispatch: 'bg-orange-100 text-orange-700',
}
