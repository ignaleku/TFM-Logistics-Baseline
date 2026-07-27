import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer,
} from 'recharts'
import { api } from '../../api'
import type { OrderSummary } from '../../types'

const MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

function Kpi({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card text-center py-4">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className="text-2xl font-bold text-slate-800">{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
    </div>
  )
}

export function DemandComplexityTab() {
  const [data, setData] = useState<OrderSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.getOrderSummary()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-center py-24 text-slate-400">Loading order distribution…</div>
  if (error) return (
    <div className="p-4 bg-red-50 rounded-xl border border-red-200 text-sm text-red-700">
      <strong>Could not load order summary.</strong> {error}
      <p className="mt-1 text-xs">Run: <code>python -m src.data.generate_orders_seasonal</code> first.</p>
    </div>
  )
  if (!data.length) return <div className="text-center py-24 text-slate-400">No order data found.</div>

  const demandData = data.map((d) => ({
    month: MONTH_ABBR[d.month - 1],
    orders: d.orders,
    urgentShare: +(d.urgent_share * 100).toFixed(1),
  }))

  const familyData = data.map((d) => ({
    month: MONTH_ABBR[d.month - 1],
    standard: +(d.pct_standard * 100).toFixed(1),
    fragile: +(d.pct_fragile * 100).toFixed(1),
    bulky: +(d.pct_bulky * 100).toFixed(1),
  }))

  const complexityData = data.map((d) => ({
    month: MONTH_ABBR[d.month - 1],
    low: +(d.pct_low * 100).toFixed(1),
    medium: +(d.pct_medium * 100).toFixed(1),
    high: +(d.pct_high * 100).toFixed(1),
  }))

  const workloadData = data.map((d) => ({
    month: MONTH_ABBR[d.month - 1],
    picking: d.avg_picking_units,
    packing: d.avg_packing_units,
    dispatch: d.avg_dispatch_units,
  }))

  const totalOrders = data.reduce((s, d) => s + d.orders, 0)
  const avgUrgent = (data.reduce((s, d) => s + d.urgent_share, 0) / data.length * 100).toFixed(1)
  const peakMonth = data.reduce((a, b) => a.orders > b.orders ? a : b)
  const valleyMonth = data.reduce((a, b) => a.orders < b.orders ? a : b)

  return (
    <div className="space-y-8">

      {/* Explanation banner */}
      <div className="p-4 bg-indigo-50 rounded-xl border border-indigo-100 text-sm text-indigo-800">
        <strong>Why this matters:</strong> Orders are not homogeneous. A <em>fragile, high-complexity</em> order
        requires 3× more packing time than a <em>standard, low-complexity</em> one — even with the same item count.
        This creates shifting bottlenecks across Picking, Packing, and Dispatch, making intelligent sequencing (RL-3)
        more valuable than a fixed policy.
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Kpi label="Total orders / year" value={totalOrders.toLocaleString()} />
        <Kpi label="Avg urgent share" value={`${avgUrgent}%`} sub="higher in Dec/Jan" />
        <Kpi label="Peak month" value={peakMonth.month_name} sub={`${peakMonth.orders.toLocaleString()} orders`} />
        <Kpi label="Valley month" value={valleyMonth.month_name} sub={`${valleyMonth.orders.toLocaleString()} orders`} />
      </div>

      {/* Demand charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Monthly Order Volume</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={demandData} margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => (v / 1000).toFixed(0) + 'k'} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: number) => v.toLocaleString()} />
              <Bar dataKey="orders" name="Orders" fill="#6366f1" radius={[4, 4, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-4">Urgent Order Share by Month (%)</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={demandData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => `${v}%`} domain={[0, 35]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => `${v}%`} />
              <Line
                type="monotone"
                dataKey="urgentShare"
                name="Urgent %"
                stroke="#f97316"
                strokeWidth={2.5}
                dot={{ r: 4, fill: '#f97316' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Product family & complexity */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-1">Product Family Distribution (%)</p>
          <p className="text-xs text-slate-400 mb-4">
            Fragile orders need 1.8× more packing time. Bulky orders need 1.6×.
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={familyData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="standard" name="Standard" fill="#6366f1" stackId="a" />
              <Bar dataKey="fragile" name="Fragile" fill="#f97316" stackId="a" />
              <Bar dataKey="bulky" name="Bulky" fill="#0ea5e9" stackId="a" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="text-sm font-semibold text-slate-600 mb-1">Complexity Distribution (%)</p>
          <p className="text-xs text-slate-400 mb-4">
            High-complexity orders need up to 1.7× more packing and 1.4× more picking time.
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={complexityData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="low" name="Low" fill="#34d399" stackId="a" />
              <Bar dataKey="medium" name="Medium" fill="#fbbf24" stackId="a" />
              <Bar dataKey="high" name="High" fill="#f87171" stackId="a" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Workload units */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-1">Average Workload Units by Month</p>
        <p className="text-xs text-slate-400 mb-4">
          Workload units are the operational load indicator passed to each simulation stage.
          Packing units are the most variable — driven by product_family × complexity_level.
          This is why the packing bottleneck shifts across months and order types.
        </p>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={workloadData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v: number) => v.toFixed(2)} />
            <Legend />
            <Line type="monotone" dataKey="picking" name="Picking units" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="packing" name="Packing units" stroke="#f97316" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="dispatch" name="Dispatch units" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Multiplier legend */}
      <div className="card">
        <p className="text-sm font-semibold text-slate-600 mb-3">Service-Time Multiplier Reference</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
          <div>
            <p className="font-semibold text-slate-700 mb-2">Picking (item-driven)</p>
            <table className="w-full text-left">
              <thead><tr className="text-slate-400"><th>Family</th><th>Cmplx</th><th>Mult</th></tr></thead>
              <tbody className="text-slate-600">
                <tr><td>Standard</td><td>Low</td><td className="font-mono">×0.9</td></tr>
                <tr><td>Fragile</td><td>Medium</td><td className="font-mono">×1.1×1.1</td></tr>
                <tr><td>Bulky</td><td>High</td><td className="font-mono">×1.3×1.4</td></tr>
              </tbody>
            </table>
          </div>
          <div>
            <p className="font-semibold text-slate-700 mb-2">Packing (most variable)</p>
            <table className="w-full text-left">
              <thead><tr className="text-slate-400"><th>Family</th><th>Cmplx</th><th>Mult</th></tr></thead>
              <tbody className="text-slate-600">
                <tr><td>Standard</td><td>Low</td><td className="font-mono">×0.8</td></tr>
                <tr><td>Fragile</td><td>Medium</td><td className="font-mono">×1.8×1.2</td></tr>
                <tr><td>Bulky</td><td>High</td><td className="font-mono">×1.6×1.7</td></tr>
              </tbody>
            </table>
          </div>
          <div>
            <p className="font-semibold text-slate-700 mb-2">Dispatch (urgency-driven)</p>
            <table className="w-full text-left">
              <thead><tr className="text-slate-400"><th>Urgency</th><th>Cmplx</th><th>Mult</th></tr></thead>
              <tbody className="text-slate-600">
                <tr><td>Normal</td><td>Low</td><td className="font-mono">×0.9</td></tr>
                <tr><td>Urgent</td><td>Medium</td><td className="font-mono">×1.3×1.1</td></tr>
                <tr><td>Urgent</td><td>High</td><td className="font-mono">×1.3×1.4</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

    </div>
  )
}
