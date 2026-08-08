import type {
  BottlenecksResponse, FilesStatus, FullResult, FuturePreview, FutureRunParams, HistoricalRunParams,
  MonthSummary, OrderSummary, PlanningProfile, RunContext, RunStartedResponse, RunStatus, UploadResponse,
} from './types'

const BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export type Mode = 'future' | 'historical'

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

  // With no mode: the static annual client-profile baseline. With a mode: the current run's
  // scoped demand summary (same as RunContext.order_summary — provided as a convenience).
  getOrderSummary: (mode?: Mode) => get<OrderSummary[]>(`/data/order-summary${mode ? `?mode=${mode}` : ''}`),

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

  runMonthlyCapacityCost: (params: HistoricalRunParams): Promise<RunStartedResponse> =>
    post('/run/monthly-capacity-cost', params),

  getRecommendations: (mode: Mode) => get<MonthSummary[]>(`/results/latest/recommendations?mode=${mode}`),

  getFullResults: (mode: Mode) => get<FullResult[]>(`/results/latest/full?mode=${mode}`),

  runStatus: () => get<RunStatus>('/run/status'),

  // ── Future planning ──────────────────────────────────────────────────────
  getPlanningProfile: () => get<PlanningProfile>('/planning/profile'),

  previewFuturePlan: (params: {
    planning_month: string
    expected_annual_orders: number
    monthly_orders_override?: number | null
    uncertainty_level: string
    hours_per_worker_month?: number | null
  }): Promise<FuturePreview> => post('/planning/preview', params),

  runFuturePlanning: (params: FutureRunParams): Promise<RunStartedResponse> =>
    post('/run/future-planning', { checkpoint: 'data/dqn_rl3_final.pt', ...params }),

  // ── Bottlenecks ───────────────────────────────────────────────────────────
  getLatestBottlenecks: (mode: Mode) => get<BottlenecksResponse>(`/results/latest/bottlenecks?mode=${mode}`),

  // ── Run context (Demand & Complexity, context banner, mode persistence) ──
  getLatestContext: (mode: Mode) => get<RunContext>(`/results/latest/context?mode=${mode}`),
}
