import type {
  BottlenecksResponse, FilesStatus, FullResult, FuturePreview, FutureRunParams, MonthSummary,
  OrderSummary, PlanningProfile, RunScope, RunStartedResponse, RunStatus, UploadResponse,
} from './types'

const BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))
  }
  return res.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const b = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail))
  }
  return res.json()
}

export const api = {
  health: () => get<{ status: string; service: string }>('/health'),

  filesStatus: () => get<FilesStatus>('/files/status'),

  getOrderSummary: () => get<OrderSummary[]>('/data/order-summary'),

  uploadOrders: async (file: File): Promise<UploadResponse> => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/upload-orders`, { method: 'POST', body: form })
    if (!res.ok) {
      const b = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail))
    }
    return res.json()
  },

  runMonthlyCapacityCost: (params: {
    orders_path: string
    checkpoint: string
    cost_late_urgent: number
    cost_late_normal: number
    worker_cost_per_hour: number
    hours_per_worker_month: number
    months?: string[] | null
  }): Promise<RunStartedResponse> => post('/run/monthly-capacity-cost', params),

  getRecommendations: () => get<MonthSummary[]>('/results/latest/recommendations'),

  getFullResults: () => get<FullResult[]>('/results/latest/full'),

  getMonthRecommendation: (monthName: string) =>
    get<MonthSummary & { min_urgent_sla_option?: FullResult; min_total_sla_option?: FullResult }>(
      `/recommend/month/${encodeURIComponent(monthName)}`
    ),

  runStatus: () => get<RunStatus>('/run/status'),

  // ── Future planning ──────────────────────────────────────────────────────
  getPlanningProfile: () => get<PlanningProfile>('/planning/profile'),

  previewFuturePlan: (params: {
    planning_month: string
    expected_annual_orders: number
    monthly_orders_override?: number | null
    uncertainty_level: string
  }): Promise<FuturePreview> => post('/planning/preview', params),

  runFuturePlanning: (params: FutureRunParams): Promise<RunStartedResponse> =>
    post('/run/future-planning', { checkpoint: 'data/dqn_rl3_final.pt', ...params }),

  // ── Bottlenecks ───────────────────────────────────────────────────────────
  getLatestBottlenecks: () => get<BottlenecksResponse>('/results/latest/bottlenecks'),

  // ── Run scope (Demand & Complexity context) ──────────────────────────────
  getLatestRunScope: () => get<RunScope>('/results/latest/run-scope'),
}
