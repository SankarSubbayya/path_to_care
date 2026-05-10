import type { Urgency } from "@/lib/types";

const STYLES: Record<
  Urgency,
  { bg: string; emoji: string; label: string; sub: string; pulse: boolean }
> = {
  red: {
    bg: "bg-gradient-to-br from-red-500 via-rose-600 to-red-700",
    emoji: "🔴",
    label: "RED — see a clinician today",
    sub: "Same-day care needed. Don't wait until tomorrow.",
    pulse: true,
  },
  yellow: {
    bg: "bg-gradient-to-br from-amber-400 via-amber-500 to-orange-500",
    emoji: "🟡",
    label: "YELLOW — see a clinician in 1–2 days",
    sub: "Not an emergency, but evaluation should not be delayed.",
    pulse: false,
  },
  green: {
    bg: "bg-gradient-to-br from-emerald-500 via-emerald-600 to-teal-600",
    emoji: "🟢",
    label: "GREEN — monitor at home",
    sub: "Watch for warning signs. Return if it gets worse.",
    pulse: false,
  },
};

export function UrgencyBadge({ urgency, escalated = false }: { urgency: Urgency; escalated?: boolean }) {
  const s = STYLES[urgency];
  return (
    <div
      className={`relative overflow-hidden rounded-2xl p-6 text-white shadow-xl ${s.bg} ${
        s.pulse ? "ptc-urgency-pulse" : ""
      }`}
    >
      {/* decorative shine */}
      <div aria-hidden className="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/10 to-white/0" />
      <div className="relative flex items-center gap-4">
        <div className="text-5xl leading-none drop-shadow-md">{s.emoji}</div>
        <div>
          <div className="text-xl font-semibold tracking-tight">{s.label}</div>
          <div className="mt-1 text-sm opacity-95">{s.sub}</div>
        </div>
      </div>
      {escalated && (
        <div className="relative mt-4 rounded-lg border border-white/30 bg-white/15 px-3 py-2 text-xs backdrop-blur-sm">
          <span className="mr-1">🛡</span>
          Safety net: model returned <span className="font-semibold">green</span>, but multiple red-flag keywords
          were detected in the narrative. Escalated to <span className="font-semibold">yellow</span> per the
          "never under-triage" rule.
        </div>
      )}
    </div>
  );
}
