"use client";

import { useState } from "react";

export interface TabSpec {
  id: string;
  label: string;
  icon?: string;
  content: React.ReactNode;
}

export function Tabs({ tabs, initial = 0 }: { tabs: TabSpec[]; initial?: number }) {
  const [active, setActive] = useState(initial);
  return (
    <div>
      <div className="flex gap-1 border-b border-gray-200">
        {tabs.map((t, i) => (
          <button
            key={t.id}
            onClick={() => setActive(i)}
            className={`px-4 py-2 text-sm font-medium transition border-b-2 -mb-px ${
              active === i
                ? "border-blue-600 text-blue-700"
                : "border-transparent text-gray-500 hover:text-gray-800"
            }`}
          >
            {t.icon && <span className="mr-1">{t.icon}</span>}
            {t.label}
          </button>
        ))}
      </div>
      <div className="pt-5">{tabs[active]?.content}</div>
    </div>
  );
}
