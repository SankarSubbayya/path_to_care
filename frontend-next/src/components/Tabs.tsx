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
      <div className="flex gap-1 rounded-2xl border border-slate-200/70 bg-white/70 p-1 backdrop-blur-sm shadow-sm">
        {tabs.map((t, i) => (
          <button
            key={t.id}
            onClick={() => setActive(i)}
            className={`flex-1 rounded-xl px-4 py-2 text-sm font-medium transition ${
              active === i
                ? "bg-gradient-to-r from-teal-600 to-cyan-600 text-white shadow-md shadow-cyan-500/20"
                : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
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
