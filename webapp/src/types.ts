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

export interface RunStatus {
  status: 'idle' | 'running' | 'completed' | 'complete' | 'failed' | 'error'
  step?: string
  progress_pct?: number
  message?: string
  error?: string | null
  started_at?: string
  updated_at?: string
}
