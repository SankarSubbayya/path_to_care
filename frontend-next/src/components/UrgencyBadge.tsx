import type { Urgency } from "@/lib/types";

const STYLES: Record<
  Urgency,
  { bg: string; ring: string; mark: string; label: string; sub: string; pulse: boolean }
> = {
  red: {
    bg: "bg-[#b14a32]",          // clay (terracotta)
    ring: "ring-[#8d3920]",
    mark: "RED",
    label: "See a clinician today",
    sub: "Same-day care needed. Don't wait until tomorrow.",
    pulse: true,
  },
  yellow: {
    bg: "bg-[#b8821e]",          // muted ochre / gold
    ring: "ring-[#946718]",
    mark: "YELLOW",
    label: "See a clinician in 1–2 days",
    sub: "Not an emergency, but evaluation should not be delayed.",
    pulse: false,
  },
  green: {
    bg: "bg-[#1f3a3d]",          // deep teal-black
    ring: "ring-[#16292b]",
    mark: "GREEN",
    label: "Monitor at home",
    sub: "Watch for warning signs. Return if it gets worse.",
    pulse: false,
  },
};

export function UrgencyBadge({ urgency, escalated = false }: { urgency: Urgency; escalated?: boolean }) {
  const s = STYLES[urgency];
  return (
    <div
      className={`relative overflow-hidden rounded-md p-6 text-[#fbf6ec] shadow-sm ring-1 ${s.bg} ${s.ring} ${
        s.pulse ? "ptc-urgency-pulse" : ""
      }`}
    >
      <div className="relative flex items-center gap-5 ptc-ui">
        <div className="border border-[#fbf6ec]/30 px-3 py-1.5 text-xs font-bold uppercase tracking-[0.2em]">
          {s.mark}
        </div>
        <div>
          <div className="text-xl font-semibold tracking-tight">{s.label}</div>
          <div className="mt-1 text-sm opacity-90">{s.sub}</div>
        </div>
      </div>
      {escalated && (
        <div className="relative mt-4 rounded border border-[#fbf6ec]/25 bg-[#fbf6ec]/10 px-3 py-2 text-xs ptc-ui">
          <span className="mr-1 font-semibold uppercase tracking-wider">Safety net</span>·
          model returned <span className="font-semibold">green</span>, but multiple red-flag keywords
          were detected. Escalated to <span className="font-semibold">yellow</span> per the
          "never under-triage" rule.
        </div>
      )}
    </div>
  );
}
