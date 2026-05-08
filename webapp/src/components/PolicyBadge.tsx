interface Props {
  policy: string
  size?: 'sm' | 'md'
}

const POLICY_MAP: Record<string, { label: string; cls: string }> = {
  fifo:         { label: 'FIFO',         cls: 'bg-slate-100 text-slate-600' },
  urgent_first: { label: 'Urgent-First', cls: 'bg-orange-100 text-orange-700' },
  rl5_dqn:      { label: 'RL-5 DQN',    cls: 'bg-violet-100 text-violet-700' },
}

export function PolicyBadge({ policy, size = 'md' }: Props) {
  const { label, cls } = POLICY_MAP[policy] ?? { label: policy, cls: 'bg-gray-100 text-gray-600' }
  const sz = size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-2.5 py-1'
  return (
    <span className={`inline-flex items-center rounded-full font-medium ${sz} ${cls}`}>
      {label}
    </span>
  )
}
