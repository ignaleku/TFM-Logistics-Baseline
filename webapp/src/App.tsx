import { useState, useEffect, useCallback } from 'react'
import { api } from './api'
import { UploadRunTab } from './components/tabs/UploadRunTab'
import { ModeResultsTab } from './components/results/ModeResultsTab'
import { MethodologyModal } from './components/MethodologyModal'

const TABS = [
  { id: 'run',        label: 'Run' },
  { id: 'future',     label: 'Future Planning' },
  { id: 'historical', label: 'Historical Analysis' },
] as const

type TabId = typeof TABS[number]['id']

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('run')
  const [apiOk, setApiOk] = useState<boolean | null>(null)
  const [methodologyOpen, setMethodologyOpen] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    api.health()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))
  }, [])

  const handleRunComplete = useCallback(() => {
    setRefreshKey((k) => k + 1)
  }, [])

  const handleViewResults = useCallback((runMode: 'future' | 'historical') => {
    setActiveTab(runMode)
  }, [])

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-screen-xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
              </svg>
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-900 leading-tight">
                Logistics Decision Support
              </h1>
              <p className="text-xs text-slate-400">
                3-stage simulation · heterogeneous orders · RL-3 sequencing · monthly operating-time capacity planning
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs">
              <span className={`w-2 h-2 rounded-full ${
                apiOk === null ? 'bg-slate-300' :
                apiOk ? 'bg-emerald-400' : 'bg-red-400'
              }`} />
              <span className="text-slate-500">
                {apiOk === null ? 'Connecting…' : apiOk ? 'Backend connected' : 'Backend offline'}
              </span>
            </div>
            <button
              onClick={() => setMethodologyOpen(true)}
              className="btn-secondary py-1.5 px-3 text-xs"
            >
              Methodology
            </button>
          </div>
        </div>

        {/* Top-level tab bar */}
        <div className="max-w-screen-xl mx-auto px-6 pb-0">
          <nav className="flex gap-1 overflow-x-auto bg-slate-100 rounded-xl p-1 mb-0 w-fit">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`tab-btn ${activeTab === t.id ? 'tab-btn-active' : 'tab-btn-inactive'}`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="h-3" />
      </header>

      {/* Main content */}
      <main className="max-w-screen-xl mx-auto px-6 py-8">
        {apiOk === false && (
          <div className="mb-6 p-4 bg-red-50 rounded-xl border border-red-200 text-sm text-red-700">
            <strong>Backend not reachable.</strong> Start it with:{' '}
            <code className="bg-red-100 px-2 py-0.5 rounded text-xs">
              python -m uvicorn src.api.main:app --reload --port 8000
            </code>
          </div>
        )}

        {activeTab === 'run' && (
          <UploadRunTab onRunComplete={handleRunComplete} onViewResults={handleViewResults} />
        )}
        {activeTab === 'future' && (
          <ModeResultsTab mode="future" refreshKey={refreshKey} onGoToRun={() => setActiveTab('run')} />
        )}
        {activeTab === 'historical' && (
          <ModeResultsTab mode="historical" refreshKey={refreshKey} onGoToRun={() => setActiveTab('run')} />
        )}
      </main>

      <MethodologyModal open={methodologyOpen} onClose={() => setMethodologyOpen(false)} />
    </div>
  )
}
