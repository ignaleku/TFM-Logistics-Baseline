interface Props {
  label: string
  onGoToRun: () => void
}

export function ResultsEmptyState({ label, onGoToRun }: Props) {
  return (
    <div className="text-center py-24">
      <p className="text-slate-400 mb-4">No {label} result is available yet.</p>
      <button className="btn-primary" onClick={onGoToRun}>Go to Run</button>
    </div>
  )
}
