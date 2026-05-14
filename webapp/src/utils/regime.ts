export interface RegimeBreakdown {
  picking: number
  packing: number
  dispatch: number
  total: number
}

const STAGES = ['picking', 'packing', 'dispatch'] as const

export function parseRegime(regime: string): RegimeBreakdown | null {
  // 3-stage format: "s321" → {picking:3, packing:2, dispatch:1}
  const match = regime?.match(/^s(\d)(\d)(\d)$/)
  if (!match) return null
  const [, p, pk, d] = match.map(Number)
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
