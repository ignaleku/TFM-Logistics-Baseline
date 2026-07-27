export function MethodTab() {
  return (
    <div className="max-w-3xl space-y-6">

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">3-Stage Warehouse Simulation</h2>
        <p className="text-sm text-slate-600 mb-3">
          The simulation models a warehouse with three sequential stages:
        </p>
        <div className="flex items-center gap-3 mb-4">
          {['Picking', 'Packing', 'Dispatch'].map((s, i) => (
            <div key={s} className="flex items-center gap-3">
              <div className="px-4 py-2 bg-indigo-600 text-white text-sm font-semibold rounded-xl">{s}</div>
              {i < 2 && <span className="text-slate-400 text-lg">→</span>}
            </div>
          ))}
        </div>
        <p className="text-sm text-slate-600">
          Each stage has configurable worker capacity. Orders flow through stages sequentially.
          Service time at each stage depends on the order's <strong>workload units</strong>, which are
          computed from product_family × complexity_level × num_items.
        </p>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">Order Heterogeneity</h2>
        <p className="text-sm text-slate-600 mb-3">
          Orders differ along three dimensions that drive different workloads at each stage:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs font-semibold text-slate-700 mb-1">Order Type</p>
            <p className="text-xs text-slate-500">Urgent (4h SLA) or Normal (24h SLA). Urgency drives dispatch workload and queue priority.</p>
          </div>
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs font-semibold text-slate-700 mb-1">Product Family</p>
            <p className="text-xs text-slate-500">Standard / Fragile / Bulky. Fragile orders require 1.8× more packing time. Bulky 1.6×.</p>
          </div>
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs font-semibold text-slate-700 mb-1">Complexity Level</p>
            <p className="text-xs text-slate-500">Low / Medium / High. High complexity multiplies packing time by 1.7× and picking by 1.4×.</p>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">Sequencing Policies</h2>
        <div className="space-y-3">
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs font-semibold text-slate-700">FIFO — First In, First Out</p>
            <p className="text-xs text-slate-500 mt-1">Orders are processed in arrival order. No priority differentiation.</p>
          </div>
          <div className="p-3 bg-orange-50 rounded-xl border border-orange-100">
            <p className="text-xs font-semibold text-orange-700">Urgent-First</p>
            <p className="text-xs text-slate-500 mt-1">Urgent orders always jump ahead of normal orders at every stage. Simple rule, often effective, but ignores downstream workload.</p>
          </div>
          <div className="p-3 bg-violet-50 rounded-xl border border-violet-100">
            <p className="text-xs font-semibold text-violet-700">RL-3 DQN — Reinforcement Learning Dynamic Policy</p>
            <p className="text-xs text-slate-500 mt-1">
              A Deep Q-Network agent makes sequencing decisions at Picking, Packing, and Dispatch.
              At each decision point the agent observes queue lengths, WIP at each stage, time elapsed,
              and the slack of the head urgent/normal order. It learns to balance urgent priority
              with overall throughput, particularly effective when bottlenecks shift due to order
              heterogeneity (fragile/complex orders clogging packing).
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">Monthly Capacity-Cost Optimisation</h2>
        <p className="text-sm text-slate-600 mb-3">
          For each month, the system evaluates 16 worker regimes × 3 policies = 48 combinations.
          The total estimated cost is:
        </p>
        <div className="p-3 bg-slate-50 rounded-xl font-mono text-xs text-slate-700 mb-3">
          Total Cost = SLA Penalty Cost + Monthly Labour Cost<br/>
          SLA Penalty = late_urgent × cost_per_late_urgent + late_normal × cost_per_late_normal<br/>
          Labour Cost = total_workers × worker_cost_per_hour × hours_per_worker_month
        </div>
        <p className="text-sm text-slate-600">
          The 16 regimes span from minimal (s111: 1-1-1 workers) to heavy (s432: 4-3-2 workers),
          covering all realistic bottleneck configurations. Regime notation is
          s{'{picking}'}{'{packing}'}{'{dispatch}'}.
        </p>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">Seasonal Demand Pattern</h2>
        <p className="text-sm text-slate-600">
          The baseline dataset has 240,000 synthetic orders across a full year with:
          <strong> peak demand</strong> in January, February, November, December;
          <strong> valley</strong> in May–August; moderate autumn ramp-up in September–October.
          Urgent order share is also seasonal: highest in December (25%) and lowest in summer (8%).
          This creates different optimal staffing configurations across months.
        </p>
      </div>

    </div>
  )
}
