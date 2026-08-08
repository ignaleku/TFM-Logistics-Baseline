import type { RunContext } from '../../types'

interface Props {
  ctx: RunContext
}

// Persistent context banner (spec §35) — comes entirely from the persisted backend manifest,
// so it survives refresh / navigation / running the other mode, never from ephemeral React state.
export function ContextBanner({ ctx }: Props) {
  const isFuture = ctx.run_mode === 'future'
  const totalOrders = (ctx.order_summary ?? []).reduce((s, r) => s + (r.orders ?? 0), 0)
  const hpm = ctx.hours_per_worker_month ?? ctx.preview?.operating_hours_per_month

  if (isFuture) {
    const p = ctx.preview
    return (
      <div className="rounded-xl bg-slate-800 text-slate-100 px-4 py-3 text-xs flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="font-semibold">Future Planning · {ctx.month_name ?? p?.month_name}</span>
        <span className="text-slate-400">·</span>
        <span>{(ctx.expected_monthly_orders ?? p?.expected_monthly_orders ?? 0).toLocaleString()} expected orders</span>
        <span className="text-slate-400">·</span>
        <span className="capitalize">{ctx.forecast_source === 'monthly_override' ? 'Monthly override' : 'Annual forecast'}</span>
        {hpm != null && (
          <>
            <span className="text-slate-400">·</span>
            <span>{hpm} operating hours</span>
          </>
        )}
        {ctx.uncertainty_level && (
          <>
            <span className="text-slate-400">·</span>
            <span className="capitalize">{ctx.uncertainty_level} uncertainty</span>
          </>
        )}
        {p?.replications != null && (
          <>
            <span className="text-slate-400">·</span>
            <span>{p.replications} scenarios</span>
          </>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-xl bg-slate-800 text-slate-100 px-4 py-3 text-xs flex flex-wrap items-center gap-x-2 gap-y-1">
      <span className="font-semibold">Historical Analysis · {ctx.month_names?.join(' + ') || '—'}</span>
      <span className="text-slate-400">·</span>
      <span>{totalOrders.toLocaleString()} historical orders</span>
      {hpm != null && (
        <>
          <span className="text-slate-400">·</span>
          <span>{hpm} operating hours per month</span>
        </>
      )}
    </div>
  )
}
