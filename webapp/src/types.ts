export interface MonthSummary {
  month: number
  month_name: string
  total_orders: number
  urgent_orders: number
  normal_orders?: number
  urgent_share: number

  // Cheapest option
  best_total_regime: string
  best_total_policy: string
  best_total_workers: number
  best_total_picking_workers?: number
  best_total_packing_workers?: number
  best_total_dispatch_workers?: number
  best_total_sla: number
  best_total_urgent_sla?: number
  best_total_normal_sla?: number
  best_total_late_cost?: number
  best_total_labour_cost?: number
  best_total_cost: number

  // Best RL-3
  best_rl3_regime?: string
  best_rl3_workers?: number
  best_rl3_sla?: number
  best_rl3_urgent_sla?: number
  best_rl3_normal_sla?: number
  best_rl3_late_cost?: number
  best_rl3_labour_cost?: number
  best_rl3_total_cost?: number
  best_rl3_gap_vs_cheapest?: number

  // Minimum workforce for urgent SLA >= 95%
  min_urgent_regime?: string
  min_urgent_policy?: string
  min_urgent_workers?: number
  min_urgent_sla?: number
  min_urgent_urgent_sla?: number
  min_urgent_normal_sla?: number
  min_urgent_late_cost?: number
  min_urgent_labour_cost?: number
  min_urgent_total_cost?: number

  // Minimum workforce for total SLA >= 80%
  min_total_sla_regime?: string
  min_total_sla_policy?: string
  min_total_sla_workers?: number
  min_total_sla?: number
  min_total_sla_urgent_sla?: number
  min_total_sla_normal_sla?: number
  min_total_sla_late_cost?: number
  min_total_sla_labour_cost?: number
  min_total_sla_total_cost?: number

  // Best urgent_first
  best_urgent_first_regime?: string
  best_urgent_first_workers?: number
  best_urgent_first_sla?: number
  best_urgent_first_urgent_sla?: number
  best_urgent_first_normal_sla?: number
  best_urgent_first_late_cost?: number
  best_urgent_first_labour_cost?: number
  best_urgent_first_total_cost?: number

  // Comparison
  rl3_minus_urgent_first_total_cost?: number

  // Labels
  cheapest_label: string
  rl3_label?: string
  min_urgent_label?: string
  min_total_sla_label?: string
  managerial_interpretation_short?: string
}

export interface FullResult {
  month: number
  month_name: string
  regime: string
  policy: string
  total_workers: number
  picking_workers?: number
  packing_workers?: number
  dispatch_workers?: number
  total_orders: number
  urgent_orders: number
  normal_orders?: number
  urgent_share: number
  total_sla: number
  urgent_sla: number
  normal_sla: number
  mean_system_time_min?: number
  p90_system_time_min?: number
  urgent_late_orders: number
  normal_late_orders: number
  estimated_late_cost: number
  estimated_worker_cost: number
  estimated_total_cost: number
  p_urgent_overall?: number | null
  p_urgent_pick?: number | null
  p_urgent_pack?: number | null
  p_urgent_dispatch?: number | null
  decisions_total?: number | null
  decisions_pick?: number | null
  decisions_pack?: number | null
  decisions_dispatch?: number | null
  feasible?: boolean
  urgent_sla_target?: number
  normal_sla_target?: number
  sla_violation?: number
  p90_total_cost?: number | null
  prob_meets_sla_targets?: number
  replication_count?: number
}

export interface OrderSummary {
  month: number
  month_name: string
  orders: number
  urgent_share: number
  mean_num_items: number
  pct_standard: number
  pct_fragile: number
  pct_bulky: number
  pct_low: number
  pct_medium: number
  pct_high: number
  avg_picking_units: number
  avg_packing_units: number
  avg_dispatch_units: number
}

export interface FilesStatus {
  uploaded_orders: boolean
  checkpoint: boolean
  latest_capacity_results: boolean
  latest_recommendations_summary: boolean
  latest_full_results: boolean
  paths: Record<string, string>
}

export interface UploadResponse {
  status: string
  total_rows: number
  date_range: string | null
  detected_months: string[]
  urgent_share: number
  message: string
}

export interface RunStartedResponse {
  status: string
  run_id: string
  message: string
}

export interface RunResponse {
  run_id: string
  status: string
  elapsed_seconds: number
  output_paths: Record<string, string>
  stdout_tail?: string
  error?: string
}

export interface RunStatusDetail {
  phase?: string
  regime?: number
  regime_total?: number
  policy?: string
  completed_simulations?: number
  estimated_total_simulations?: number
  finalist?: number
  finalist_total?: number
  replication?: number
  replication_total?: number
  candidate?: string
  iteration?: number
  iteration_total?: number
}

export interface RunStatus {
  status: 'idle' | 'running' | 'completed' | 'complete' | 'failed' | 'error'
  step?: string
  progress_pct?: number
  message?: string
  error?: string | null
  started_at?: string
  updated_at?: string
  run_mode?: 'historical' | 'future'
  detail?: RunStatusDetail
}

// ── Future planning ──────────────────────────────────────────────────────────

export interface PlanningProfile {
  version: number
  months: { number: number; name: string }[]
  sla_targets: { urgent_target: number; normal_target: number }
  uncertainty_levels: { level: string; demand_cv: number; arrival_cv: number; description: string }[]
  cost_defaults: {
    urgent_late_cost: number
    normal_late_cost: number
    worker_cost_per_hour: number
    hours_per_worker_month: number
  }
  default_replications: number
  regimes: string[]
}

export interface FuturePreview {
  month: number
  month_name: string
  expected_monthly_orders: number
  source: 'annual_forecast' | 'monthly_override'
  annual_share: number
  urgent_share: number
  expected_avg_items: number
  product_family_shares: Record<string, number>
  complexity_shares: Record<string, number>
  operating_days: number
  operating_hours_per_day: number
  expected_orders_per_operating_hour: number
  uncertainty_level: string
  uncertainty_assumptions: { demand_cv: number; arrival_cv: number }
  sla_targets: { urgent_target: number; normal_target: number }
  replications: number
}

export interface FutureRunParams {
  planning_month: string
  expected_annual_orders: number
  monthly_orders_override?: number | null
  uncertainty_level: string
  checkpoint?: string
  cost_late_urgent: number
  cost_late_normal: number
  worker_cost_per_hour: number
  hours_per_worker_month: number
  current_picking_workers?: number | null
  current_packing_workers?: number | null
  current_dispatch_workers?: number | null
}

// ── Bottleneck / capacity ────────────────────────────────────────────────────

export interface StageBottleneck {
  stage: 'picking' | 'packing' | 'dispatch'
  stage_label: string
  pressure_score: number
  utilisation_component: number
  wait_component: number
  late_wait_component: number
  queue_component: number
  utilisation: number
  p95_wait_min: number
  avg_wait_min: number
  avg_queue_len: number
  max_queue_len: number
  late_wait_share: number
  rank: number
  is_primary_bottleneck: boolean
  explanation?: string
}

export interface BreakEven {
  worker_monthly_cost: number
  urgent_only_break_even_orders: number | null
  normal_only_break_even_orders: number | null
  mixed_break_even_orders: number | null
  current_avg_penalty_per_late_order: number
}

export interface AdaptiveTrailEntry {
  iteration: number
  parent_regime: string
  candidate_regime: string
  added_stage: string
  policy: string
  labour_cost_increase: number
  late_penalty_reduction: number
  total_cost_diff: number
  urgent_sla_before: number
  urgent_sla_after: number
  normal_sla_before: number
  normal_sla_after: number
  overall_sla_before: number
  overall_sla_after: number
  bottleneck_before: string
  accepted: boolean
  reason: string
}

export interface PolicyComparisonEntry {
  policy: string
  total_cost: number
  total_sla: number
  urgent_sla: number
  normal_sla: number
  urgent_late_orders: number
  normal_late_orders: number
  late_orders: number
  feasible: boolean
  sla_violation: number
  picking_workers: number
  packing_workers: number
  dispatch_workers: number
  starvation_pattern: boolean
}

export interface CapacityLevelDiagnostics {
  base_regimes_tested: number
  feasible_by_policy: Record<string, { feasible_count: number; tested_count: number }>
  adaptive_candidates_tested: number
  adaptive_candidates_accepted: number
}

export interface SelectedRecommendation {
  regime: string
  policy: string
  picking_workers: number
  packing_workers: number
  dispatch_workers: number
  total_sla: number
  urgent_sla: number
  normal_sla: number
  estimated_total_cost: number
  feasible: boolean
  sla_violation: number
  regime_source: 'base' | 'adaptive'
}

export interface MonthBottleneckReport {
  run_mode: 'historical' | 'future'
  month: number
  month_name: string
  selected_recommendation: SelectedRecommendation
  sla_targets: { urgent_target: number; normal_target: number }
  bottleneck_ranking: StageBottleneck[]
  primary_bottleneck: string
  break_even: BreakEven
  adaptive_search: {
    triggered: boolean
    trail?: AdaptiveTrailEntry[]
    stop_reason?: string
    final_regime?: string
    final_policy?: string
    regime_changed?: boolean
    simulations_executed?: number
  }
  explanation: string
  scenario_preview?: FuturePreview
  replication_count?: number
  policy_comparison: PolicyComparisonEntry[]
  recommended_policy: string
  capacity_level_diagnostics: CapacityLevelDiagnostics
}

export interface BottlenecksResponse {
  run_mode: 'historical' | 'future'
  months: MonthBottleneckReport[]
}

// ── Run scope (Demand & Complexity context) ─────────────────────────────────

export interface RunScopeOrderSummaryRow {
  month: number
  month_name: string
  orders: number
  urgent_share: number
  mean_num_items: number
  pct_standard: number
  pct_fragile: number
  pct_bulky: number
  pct_low: number
  pct_medium: number
  pct_high: number
  avg_picking_units: number
  avg_packing_units: number
  avg_dispatch_units: number
}

export interface RunScope {
  run_mode: 'historical' | 'future'
  generated_at: string
  months: number[]
  month_names: string[]
  order_summary: RunScopeOrderSummaryRow[]
  planning_month?: number
  month_name?: string
  forecast_source?: 'annual_forecast' | 'monthly_override'
  uncertainty_level?: string
  expected_monthly_orders?: number
  preview?: FuturePreview
}
