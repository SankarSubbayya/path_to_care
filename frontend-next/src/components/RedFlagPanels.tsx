// Side-by-side: rule-based cross-check (string-match in the narrative) vs.
// the triage reasoner model's red_flags_noted. Intentional duplication — the
// rule check is the safety net the model can't fail silently against.

export function RedFlagPanels({
  ruleBased,
  modelNoted,
}: {
  ruleBased: string[];
  modelNoted: string[];
}) {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-amber-700">
          Rule-based cross-check
        </div>
        <div className="mt-2 text-sm text-amber-900">
          {ruleBased.length === 0 ? (
            <span className="italic text-amber-600">No red-flag keywords detected.</span>
          ) : (
            <ul className="list-disc list-inside">
              {ruleBased.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          )}
        </div>
      </div>
      <div className="rounded-xl border border-rose-200 bg-rose-50 p-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-rose-700">
          Model — red flags noted
        </div>
        <div className="mt-2 text-sm text-rose-900">
          {modelNoted.length === 0 ? (
            <span className="italic text-rose-600">none</span>
          ) : (
            <ul className="list-disc list-inside">
              {modelNoted.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
