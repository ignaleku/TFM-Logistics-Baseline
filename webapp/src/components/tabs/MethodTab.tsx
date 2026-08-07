export function MethodTab() {
  return (
    <div className="max-w-3xl space-y-6">

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">Historical Analysis vs. Future Planning</h2>
        <p className="text-sm text-slate-600 mb-3">
          This is a prescriptive planning tool with two complementary modes that share the same simulator:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs font-semibold text-slate-700 mb-1">Historical Analysis</p>
            <p className="text-xs text-slate-500">
              Upload a real order-level CSV from a past period for counterfactual analysis: which policy and
              workforce would have worked best, and where the bottleneck was.
            </p>
          </div>
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs font-semibold text-slate-700 mb-1">Future Planning</p>
            <p className="text-xs text-slate-500">
              Provide a small set of aggregate inputs (planning month, expected annual orders, uncertainty).
              The system does <strong>not</strong> predict every future order — it transforms the forecast into
              simulated operational scenarios via the configured client planning profile, then prescribes
              workforce capacity and policy. Results are scenario-based estimates, not guarantees.
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">Future Planning: Screening + Validation</h2>
        <p className="text-sm text-slate-600 mb-3">
          Running every workforce configuration three full times is unnecessarily slow, since most
          configurations are obviously uncompetitive after a single demand scenario. Instead:
        </p>
        <ol className="list-decimal list-inside space-y-1.5 text-sm text-slate-600 mb-3">
          <li>All base workforce configurations are <strong>screened under one common scenario</strong> (replication #1).</li>
          <li>The four most promising configurations are <strong>validated under two additional scenarios</strong> (replications #2, #3).</li>
          <li>The final recommendation always uses the <strong>three-scenario aggregated metrics</strong> for those validated
            finalists — never a single-scenario screening result.</li>
          <li>Adaptive capacity candidates follow the same principle: each is screened on one scenario against the
            current (already validated) recommendation, and only validated on the remaining scenarios if it looks
            competitive — so a promising candidate is never accepted on the strength of a single lucky scenario.</li>
        </ol>
        <p className="text-sm text-slate-600">
          This keeps three-scenario robustness for every configuration that could plausibly be recommended, while
          skipping full replication for configurations already ruled out by the screening pass.
        </p>
      </div>

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
              the slack of the head urgent/normal order, and the current worker count per stage (added after
              an audit found the agent previously could not tell a low-capacity regime from a high-capacity
              one). Not assumed to always win — see Policy Comparison for feasibility and starvation checks.
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">Fair Policy Comparison</h2>
        <p className="text-sm text-slate-600 mb-3">
          FIFO, Urgent-First and RL-3 DQN are compared under <strong>common random numbers</strong>: for the same
          scenario seed, every order gets an identical service-time draw at every stage regardless of which
          policy processes it, sampled once up front rather than per-policy. Differences in outcome therefore
          reflect the sequencing decision, not random luck.
        </p>
        <p className="text-sm text-slate-600">
          The Policy Comparison tab's primary view holds workforce capacity fixed at the <strong>final recommended
          regime</strong> and compares the three policies only there — not averaged across every tested capacity
          level, which would mix in obviously undersized or oversized configurations that were never going to be
          recommended. A secondary "Performance Across Tested Capacity Levels" section keeps that broader view
          available for diagnostics.
        </p>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">Bottleneck Detection &amp; Adaptive Capacity</h2>
        <p className="text-sm text-slate-600 mb-2">
          Each stage is scored on utilisation, p95 wait, share of late-order waiting, and average queue length
          into one transparent pressure score, ranking Picking / Packing / Dispatch. When the recommended
          configuration doesn't meet SLA targets (or is near capacity with real late-order cost), an adaptive
          search adds one worker to the top bottleneck stage(s) and re-evaluates all three policies — the best
          policy can change as capacity changes. Every tested candidate is logged as accepted or rejected with
          a reason.
        </p>
        <p className="text-sm text-slate-600">
          Extra-worker economics: a worker's monthly cost is compared against the theoretical late-order
          break-even and the actual simulated marginal impact of adding them.
        </p>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">SLA Feasibility</h2>
        <p className="text-sm text-slate-600">
          A configuration is only recommended as "cheapest" if it meets both SLA floors (urgent and normal).
          If no base regime is feasible, the system labels its pick "best available; SLA targets not fully
          met" rather than presenting a pathological result (e.g. 100% urgent / ~2% normal) as a winner.
        </p>
      </div>

      <div className="card">
        <h2 className="text-base font-bold text-slate-800 mb-3">RL-3 Audit &amp; Generalisation</h2>
        <p className="text-sm text-slate-600">
          A suspicious December result (RL-3 near-perfect on urgent SLA but collapsing on normal SLA) was
          audited before any retraining: urgent-first's queue logic was validated, the state was checked for
          future-information leakage (none found), and RL-3's behaviour was found to be inconsistent across
          workforce regimes — evidence of a generalisation gap rather than a simple reward bug alone. The fix
          added capacity features to the state, widened training to a stratified 12-regime mix, and modestly
          rebalanced the reward. RL-3 is evaluated on both the regimes it trained on and 4 exact-held-out
          regimes to characterise how it generalises, not just whether it "wins".
        </p>
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
