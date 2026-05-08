export function fmtEuro(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '—'
  return new Intl.NumberFormat('en-EU', {
    style: 'currency',
    currency: 'EUR',
    maximumFractionDigits: 0,
  }).format(value)
}

export function fmtPct(value: number | null | undefined, decimals = 1): string {
  if (value == null || isNaN(value)) return '—'
  return `${(value * 100).toFixed(decimals)}%`
}

export function fmtNum(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '—'
  return Math.round(value).toString()
}

export function fmtDelta(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '—'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${fmtEuro(value)}`
}
