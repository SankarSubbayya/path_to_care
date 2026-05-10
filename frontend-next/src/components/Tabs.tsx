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
      <div
        className="flex gap-0 border-b border-[var(--rule)] ptc-ui"
        role="tablist"
      >
        {tabs.map((t, i) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={active === i}
            onClick={() => setActive(i)}
            className={`-mb-px border-b-2 px-5 py-2.5 text-sm font-medium uppercase tracking-wider transition ${
              active === i
                ? "border-[var(--clay)] text-[var(--ink)]"
                : "border-transparent text-[var(--ink-muted)] hover:text-[var(--ink)]"
            }`}
          >
            {t.icon && <span className="mr-1.5">{t.icon}</span>}
            {t.label}
          </button>
        ))}
      </div>
      <div className="pt-6">{tabs[active]?.content}</div>
    </div>
  );
}
