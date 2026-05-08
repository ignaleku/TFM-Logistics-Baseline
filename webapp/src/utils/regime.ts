export interface RegimeBreakdown {
  picking: number
  qc: number
  packing: number
  labelling: number
  dispatch: number
  total: number
}

const STAGES = ['picking', 'qc', 'packing', 'labelling', 'dispatch'] as const

export function parseRegime(regime: string): RegimeBreakdown | null {
  // Parses e.g. "s32211" → {picking:3, qc:2, packing:2, labelling:1, dispatch:1}
  const match = regime?.match(/^s(\d)(\d)(\d)(\d)(\d)$/)
  if (!match) return null
  const [, p, q, pk, l, d] = match.map(Number)
  return {
    picking: p,
    qc: q,
    packing: pk,
    labelling: l,
    dispatch: d,
    total: p + q + pk + l + d,
  }
}

export const STAGE_LABELS: Record<typeof STAGES[number], string> = {
  picking: 'Picking',
  qc: 'QC',
  packing: 'Packing',
  labelling: 'Labelling',
  dispatch: 'Dispatch',
}

export const STAGE_COLORS: Record<typeof STAGES[number], string> = {
  picking: 'bg-indigo-100 text-indigo-700',
  qc: 'bg-violet-100 text-violet-700',
  packing: 'bg-sky-100 text-sky-700',
  labelling: 'bg-emerald-100 text-emerald-700',
  dispatch: 'bg-orange-100 text-orange-700',
}
