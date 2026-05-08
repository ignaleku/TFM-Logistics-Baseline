export interface MonthSummary {
  month: number
  month_name: string
  total_orders: number
  urgent_orders: number
  urgent_share: number

  best_total_regime: string
  best_total_policy: string
  best_total_strategy_label: string
  best_total_workers: number
  best_total_cost: number
  best_total_sla: number

  best_sla_regime: string
  best_sla_policy: string
  best_sla_value: number
  best_sla_workers: number

  min_workers_regime: string
  min_workers_policy: string
  min_workers_count: number
  min_workers_sla: number

  best_rl5_regime: string
  best_rl5_workers: number
  best_rl5_sla: number
  best_rl5_urgent_sla: number
  best_rl5_normal_sla: number
  best_rl5_late_cost: number
  best_rl5_labour_cost: number
  best_rl5_total_cost: number
  best_rl5_gap_vs_cheapest: number

  balanced_regime: string
  balanced_policy: string
  balanced_workers: number
  balanced_sla: number
  balanced_urgent_sla: number
  balanced_normal_sla: number
  balanced_late_cost: number
  balanced_labour_cost: number
  balanced_total_cost: number

  best_under_budget_regime: string
  best_under_budget_policy: string
  best_under_budget_workers: number
  best_under_budget_sla: number
  best_under_budget_urgent_sla: number
  best_under_budget_normal_sla: number
  best_under_budget_total_cost: number

  cheapest_label: string
  rl5_label: string
  balanced_label: string
  under_budget_label: string
  managerial_interpretation_short: string

  rl5_vs_best_sla_diff: number
  rl5_vs_best_cost_diff: number
}

export interface FullResult {
  month: number
  month_name: string
  regime: string
  policy: string
  total_workers: number
  picking_workers?: number
  quality_check_workers?: number
  packing_workers?: number
  labelling_workers?: number
  dispatch_workers?: number
  total_orders: number
  urgent_orders: number
  normal_orders: number
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
  p_urgent_quality_check?: number | null
  p_urgent_pack?: number | null
  p_urgent_labelling?: number | null
  p_urgent_dispatch?: number | null
  decisions_total?: number | null
  decisions_pick?: number | null
  decisions_quality_check?: number | null
  decisions_pack?: number | null
  decisions_labelling?: number | null
  decisions_dispatch?: number | null
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

export interface RunResponse {
  run_id: string
  status: string
  elapsed_seconds: number
  output_paths: Record<string, string>
  stdout_tail?: string
  error?: string
}
