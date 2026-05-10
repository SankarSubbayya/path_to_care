import type { ConditionGuess } from "@/lib/types";

export function ConditionList({ items }: { items: ConditionGuess[] }) {
  if (!items || items.length === 0) {
    return (
      <p className="text-sm text-gray-500 italic">
        No top-3 candidates were produced for this image.
      </p>
    );
  }
  return (
    <ul className="space-y-2">
      {items.map((it, i) => (
        <li key={i} className="rounded-lg border border-gray-200 bg-white p-3">
          <div className="flex items-center justify-between gap-3">
            <div className="font-medium text-gray-900">{it.condition}</div>
            <div className="font-mono text-sm text-gray-500">
              {(it.confidence * 100).toFixed(0)}%
            </div>
          </div>
          <div className="mt-2 h-2 w-full rounded-full bg-gray-100">
            <div
              className="h-2 rounded-full bg-blue-500"
              style={{ width: `${Math.max(2, it.confidence * 100)}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}
