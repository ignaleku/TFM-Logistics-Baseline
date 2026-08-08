import { MethodTab } from './tabs/MethodTab'

interface Props {
  open: boolean
  onClose: () => void
}

export function MethodologyModal({ open, onClose }: Props) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-6" onClick={onClose}>
      <div
        className="bg-slate-50 rounded-2xl shadow-xl max-w-4xl w-full my-8 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-bold text-slate-800">Methodology</h2>
          <button
            className="text-slate-400 hover:text-slate-600 text-xl leading-none px-2"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <MethodTab />
      </div>
    </div>
  )
}
