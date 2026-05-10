import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { FullResult } from '../../types'
import { fmtPct } from '../../utils/format'

interface Props {
  results: FullResult[]
}

const STAGES = ['pick', 'pack', 'dispatch'] as const
const STAGE_LABELS: Record<string, string> = {
  pick:     'Picking',
  pack:     'Packing',
  dispatch: 'Dispatch',
}
const STAGE_COLORS = ['#6366f1', '#0ea5e9', '#f97316']

function avg(arr: number[]) {
  const valid = arr.filter((v) => v != null && !isNaN(v))
  return valid.length ? valid.reduce((s, v) => s + v, 0) / valid.length : null
}

export function RL3PolicyTab({ results }: Props) {
  const rl3 = useMemo(() => results.filter((r) => r.policy === 'rl3_dqn'), [results])

  const decisionRates = useMemo(() => {
    if (!rl3.length) return null
    return STAGES.map((s) => {
      const pKey = `p_urgent_${s}` as keyof FullResult
      const dKey = `decisions_${s}` as keyof FullResult
      return {
        stage: STAGE_LABELS[s],
        p_urgent: avg(rl3.map((r) => (r[pKey] as number | null | undefined) ?? NaN)),
        decisions: avg(rl3.map((r) => (r[dKey] as number | null | undefined) ?? NaN)),
      }
    })
  }, [rl3])

  const hasDecisionData = decisionRates?.some((d) => d.p_urgent != null)

  return (
    <div className="space-y-8 max-w-4xl">
      {/* What is RL-3 */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-4">What is RL-3?</p>
        <p className="text-sm text-slate-600 mb-4">
          <strong>RL-3 does not change the physical process.</strong> It changes the
          priority rule used when a worker becomes available at Picking, Packing, or Dispatch.
        </p>

        {/* State → DQN → Action diagram */}
        <div className="flex flex-col md:flex-row items-stretch gap-4">
          <div className="flex-1 bg-indigo-50 rounded-xl p-4 border border-indigo-100">
            <p className="text-xs font-semibold text-indigo-600 uppercase tracking-wide mb-3">State Inputs</p>
            <ul className="space-y-1.5 text-sm text-indigo-800">
              {[
                'Queue lengths at Picking, Packing, Dispatch',
                'Work-in-progress per stage',
                'Simulation time',
                'SLA slack (remaining time)',
                'Current stage worker is at',
              ].map((s) => (
                <li key={s} className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0" />
                  {s}
                </li>
              ))}
            </ul>
          </div>

          <div className="flex items-center justify-center px-4">
            <div className="flex flex-col items-center gap-2">
              <span className="text-2xl text-slate-300">→</span>
              <div className="bg-violet-600 text-white text-sm font-bold px-5 py-3 rounded-xl text-center shadow-md">
                DQN<br />
                <span className="text-violet-200 text-xs font-normal">Neural Network</span>
              </div>
              <span className="text-2xl text-slate-300">→</span>
            </div>
          </div>

          <div className="flex-1 bg-emerald-50 rounded-xl p-4 border border-emerald-100">
            <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide mb-3">Actions</p>
            <div className="space-y-2">
              <div className="flex items-center gap-3 p-3 bg-white rounded-lg border border-emerald-100">
                <span className="text-lg">🔴</span>
                <div>
                  <p className="text-sm font-semibold text-slate-800">Pick Urgent</p>
                  <p className="text-xs text-slate-500">Prioritise the next urgent order</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-white rounded-lg border border-emerald-100">
                <span className="text-lg">🔵</span>
                <div>
                  <p className="text-sm font-semibold text-slate-800">Pick Normal</p>
                  <p className="text-xs text-slate-500">Prioritise the next normal order</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-500">
          RL-3 does not change the physical process. It changes the priority rule used when
          a worker becomes available at any of the three stages.
        </div>
      </div>

      {/* Process flow */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-4">3-Stage Process Flow</p>
        <div className="flex items-center gap-2 flex-wrap">
          {[
            { label: 'Picking', color: 'bg-indigo-100 text-indigo-700 border-indigo-200' },
            { label: '→', color: '' },
            { label: 'Packing', color: 'bg-sky-100 text-sky-700 border-sky-200' },
            { label: '→', color: '' },
            { label: 'Dispatch', color: 'bg-orange-100 text-orange-700 border-orange-200' },
          ].map((item, i) =>
            item.color ? (
              <span key={i} className={`px-4 py-2 rounded-xl border text-sm font-semibold ${item.color}`}>
                {item.label}
              </span>
            ) : (
              <span key={i} className="text-slate-400 text-lg font-bold">{item.label}</span>
            )
          )}
        </div>
        <p className="mt-3 text-xs text-slate-400">
          RL-3 acts at each stage when a worker becomes free and both urgent and normal orders are waiting.
        </p>
      </div>

      {/* Checkpoint info */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-3">Model Details</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="stat-label">Architecture</p>
            <p className="font-medium text-slate-700">DQN (Deep Q-Network)</p>
          </div>
          <div>
            <p className="stat-label">Checkpoint</p>
            <code className="text-xs text-slate-600">dqn_rl3_final.pt</code>
          </div>
          <div>
            <p className="stat-label">Actions</p>
            <p className="font-medium text-slate-700">2 (urgent / normal)</p>
          </div>
          <div>
            <p className="stat-label">Stages</p>
            <p className="font-medium text-slate-700">3 (Picking, Packing, Dispatch)</p>
          </div>
        </div>
      </div>

      {/* Decision rates */}
      {hasDecisionData && decisionRates ? (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="card">
            <p className="text-sm font-semibold text-slate-600 mb-4">
              Avg Urgent Decision Rate by Stage (RL-3 only)
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={decisionRates.map((d, i) => ({ ...d, fill: STAGE_COLORS[i] }))}
                margin={{ top: 4, right: 16, bottom: 4, left: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="stage" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v) => fmtPct(v)} domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip formatter={(v: number) => fmtPct(v)} />
                <Bar dataKey="p_urgent" name="P(urgent decision)" fill="#7c3aed" radius={[4, 4, 4, 4]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <p className="text-sm font-semibold text-slate-600 mb-4">
              Avg Decisions per Stage
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={decisionRates}
                margin={{ top: 4, right: 16, bottom: 4, left: 16 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="stage" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="decisions" name="Avg Decisions" fill="#a78bfa" radius={[4, 4, 4, 4]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : rl3.length > 0 ? (
        <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-sm text-slate-500">
          Decision rate columns (p_urgent_pick, decisions_pick, …) not found in results.
          These are populated when the full results CSV includes RL-3 decision metrics.
        </div>
      ) : (
        <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-sm text-slate-500">
          No RL-3 results available. Run a simulation to see decision rates.
        </div>
      )}
    </div>
  )
}
