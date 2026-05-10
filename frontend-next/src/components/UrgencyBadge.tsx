import type { Urgency } from "@/lib/types";

const STYLES: Record<Urgency, { bg: string; ring: string; emoji: string; label: string; sub: string }> = {
  red: {
    bg: "bg-red-600",
    ring: "ring-red-200",
    emoji: "🔴",
    label: "RED — see a clinician today",
    sub: "Same-day care needed. Don't wait until tomorrow.",
  },
  yellow: {
    bg: "bg-amber-500",
    ring: "ring-amber-200",
    emoji: "🟡",
    label: "YELLOW — see a clinician in 1–2 days",
    sub: "Not an emergency, but evaluation should not be delayed.",
  },
  green: {
    bg: "bg-emerald-600",
    ring: "ring-emerald-200",
    emoji: "🟢",
    label: "GREEN — monitor at home",
    sub: "Watch for warning signs. Return if it gets worse.",
  },
};

export function UrgencyBadge({ urgency, escalated = false }: { urgency: Urgency; escalated?: boolean }) {
  const s = STYLES[urgency];
  return (
    <div className={`rounded-2xl p-6 text-white shadow-lg ring-4 ${s.bg} ${s.ring}`}>
      <div className="flex items-center gap-4">
        <div className="text-5xl leading-none">{s.emoji}</div>
        <div>
          <div className="text-xl font-semibold">{s.label}</div>
          <div className="mt-1 text-sm opacity-90">{s.sub}</div>
        </div>
      </div>
      {escalated && (
        <div className="mt-4 rounded-md bg-white/20 px-3 py-2 text-xs">
          🛡 Safety net: model returned <span className="font-semibold">green</span>, but multiple red-flag keywords
          were detected in the narrative. Escalated to <span className="font-semibold">yellow</span> per the
          “never under-triage” rule.
        </div>
      )}
    </div>
  );
}
