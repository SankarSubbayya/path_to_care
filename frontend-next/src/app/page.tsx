"use client";

import { useState } from "react";
import { InputForm } from "@/components/InputForm";
import { Tabs } from "@/components/Tabs";
import { UrgencyBadge } from "@/components/UrgencyBadge";
import { ConditionList } from "@/components/ConditionList";
import { SoapCard } from "@/components/SoapCard";
import { RedFlagPanels } from "@/components/RedFlagPanels";
import type { TriageResult } from "@/lib/types";

const PATIENT_DISCLAIMER =
  "This is not a diagnosis. The system gives decision-support only. " +
  "Final treatment decisions belong to a licensed clinician. " +
  "If symptoms get worse — spreading redness, high fever, severe pain, breathing trouble — go to the clinic right away.";

const CLINICIAN_DISCLAIMER =
  "Decision-support output produced by Gemma 4 31B-it (multimodal) hosted via vLLM on AMD MI300X. " +
  "Top-3 image differential always shown with confidence — single-class output is impossible by construction. " +
  "Cardinal-rule rewriter applied to all patient-facing text. Use as a triage hand-off, not a diagnosis.";

function PatientView({ r }: { r: TriageResult }) {
  return (
    <div className="space-y-5">
      <UrgencyBadge urgency={r.urgency} escalated={r.safety_escalation} />

      <section className="ptc-card rounded-2xl p-5">
        <div className="flex items-center gap-2">
          <span className="text-xl">🩹</span>
          <h3 className="text-base font-semibold text-slate-900">What this likely is</h3>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-700">{r.reasoning}</p>
      </section>

      <section className="ptc-card rounded-2xl p-5">
        <div className="flex items-center gap-2">
          <span className="text-xl">💸</span>
          <h3 className="text-base font-semibold text-slate-900">Should you go to the clinic?</h3>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-700">{r.patient_framing}</p>
        <p className="mt-3 rounded-lg border border-cyan-100 bg-cyan-50/80 p-3 text-xs text-cyan-900">
          <span className="mr-1">📍</span>{r.village.blurb}
        </p>
      </section>

      <section className="ptc-card rounded-2xl p-5">
        <div className="flex items-center gap-2">
          <span className="text-xl">⚠️</span>
          <h3 className="text-base font-semibold text-slate-900">Watch for these signs</h3>
        </div>
        <p className="mt-1 text-xs text-slate-500">If any of these happen, return immediately.</p>
        <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {(r.red_flags_noted.length > 0
            ? r.red_flags_noted
            : ["spreading redness", "high fever", "severe pain", "breathing trouble"]
          ).map((f, i) => (
            <li key={i} className="flex items-start gap-2 rounded-lg bg-rose-50/80 px-3 py-2 text-sm text-rose-900">
              <span className="mt-0.5 text-rose-500">●</span>
              <span>{f}</span>
            </li>
          ))}
        </ul>
      </section>

      <p className="rounded-lg border border-slate-200/60 bg-slate-100/70 p-3 text-xs text-slate-600">
        <span className="mr-1">ℹ️</span>{PATIENT_DISCLAIMER}
      </p>
    </div>
  );
}

function ClinicianView({ r }: { r: TriageResult }) {
  const urgColor =
    r.urgency === "red"
      ? "from-red-500 to-rose-600"
      : r.urgency === "yellow"
        ? "from-amber-400 to-orange-500"
        : "from-emerald-500 to-teal-600";
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-700">
            <span>🖼️</span> Image — top-3 differential
          </h3>
          <ConditionList items={r.image_top3} />
        </div>
        <div className="lg:col-span-2">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-700">
            <span>📋</span> Pre-visit SOAP
          </h3>
          <SoapCard soap={r.soap} />
        </div>
      </div>

      <div>
        <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-700">
          <span>🚩</span> Red flags — rule-based vs. model
        </h3>
        <RedFlagPanels
          ruleBased={r.cross_check_red_flags}
          modelNoted={r.red_flags_noted}
        />
      </div>

      <div className="ptc-card rounded-2xl p-5">
        <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-700">
          <span>⚖️</span> Triage
        </h3>
        <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className={`rounded-xl bg-gradient-to-br ${urgColor} p-4 text-white shadow-md`}>
            <div className="text-xs font-medium uppercase tracking-wide opacity-90">Urgency</div>
            <div className="mt-1 text-3xl font-bold uppercase tracking-wide">{r.urgency}</div>
            {r.safety_escalation && (
              <div className="mt-1 text-xs opacity-95">↑ escalated by safety net</div>
            )}
          </div>
          <div className="md:col-span-2">
            <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Reasoning</div>
            <p className="mt-1 text-sm leading-relaxed text-slate-800">{r.reasoning}</p>
          </div>
        </div>
      </div>

      <p className="rounded-lg border border-slate-200/60 bg-slate-100/70 p-3 text-xs text-slate-600">
        <span className="mr-1">ℹ️</span>{CLINICIAN_DISCLAIMER}
      </p>
    </div>
  );
}

function AuditView({ r }: { r: TriageResult }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Wall time" value={`${r.wall_seconds.toFixed(1)} s`} accent="cyan" />
        <Stat label="Parse OK" value={r.parse_ok ? "yes" : "no"} accent={r.parse_ok ? "green" : "red"} />
        <Stat label="Cardinal-rule rewrites" value={String(r.cardinal_rule_rewrites)} accent="amber" />
        <Stat label="Safety escalation" value={r.safety_escalation ? "yes" : "no"} accent={r.safety_escalation ? "amber" : "slate"} />
      </div>
      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-700">
          MCP tool invocations
        </h3>
        {r.tool_invocations && r.tool_invocations.length > 0 ? (
          <div className="space-y-2">
            {r.tool_invocations.map((t, i) => (
              <div key={i} className="rounded-md border border-gray-200 bg-white p-3">
                <div className="flex items-center gap-2">
                  <span className={t.ok ? "text-green-700" : "text-red-700"}>{t.ok ? "✓" : "✗"}</span>
                  <code className="text-xs font-semibold text-gray-900">{t.name}</code>
                </div>
                <pre className="mt-1 overflow-auto text-xs text-gray-700">{JSON.stringify(t.meta, null, 2)}</pre>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-500">no MCP tools invoked this turn (text-only path)</p>
        )}
      </div>
      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-700">
          Cross-check red flags (rule-based)
        </h3>
        <code className="block whitespace-pre rounded-md bg-gray-50 p-3 text-xs text-gray-800">
          {JSON.stringify(r.cross_check_red_flags, null, 2)}
        </code>
      </div>
      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-700">
          Raw model output (after cardinal-rule rewriter)
        </h3>
        <pre className="max-h-80 overflow-auto rounded-md bg-gray-50 p-3 text-xs text-gray-800">
          {r.raw_model_output || "(empty)"}
        </pre>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  accent = "slate",
}: {
  label: string;
  value: string;
  accent?: "slate" | "cyan" | "green" | "red" | "amber";
}) {
  const accentMap: Record<string, string> = {
    slate: "before:bg-slate-300",
    cyan: "before:bg-cyan-500",
    green: "before:bg-emerald-500",
    red: "before:bg-rose-500",
    amber: "before:bg-amber-500",
  };
  return (
    <div
      className={`relative overflow-hidden rounded-xl border border-slate-200/70 bg-white/80 p-3 shadow-sm transition hover:shadow-md before:absolute before:left-0 before:top-0 before:h-full before:w-1 ${accentMap[accent]}`}
    >
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 font-mono text-sm font-semibold text-slate-900">{value}</div>
    </div>
  );
}

export default function Home() {
  const [result, setResult] = useState<TriageResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  return (
    <main className="mx-auto max-w-5xl px-4 pb-16 pt-10">
      <header className="mb-8">
        <div className="flex flex-wrap items-center gap-2">
          <span className="ptc-pill"><span className="dot" />vLLM live</span>
          <span className="ptc-pill">⚡ AMD MI300X · 192 GB</span>
          <span className="ptc-pill">🧠 Gemma 4 31B-it</span>
          <span className="ptc-pill">🔬 SCIN top-16 LoRA · +7.0 pp</span>
          <a
            href="https://github.com/SankarSubbayya/path_to_care"
            className="ptc-pill hover:bg-white"
            target="_blank"
            rel="noreferrer"
          >
            <span>★</span> GitHub
          </a>
        </div>

        <div className="mt-5 flex items-start gap-4">
          <div className="flex h-14 w-14 flex-none items-center justify-center rounded-2xl bg-gradient-to-br from-teal-500 via-cyan-500 to-blue-600 text-3xl shadow-lg shadow-cyan-500/30">
            🩺
          </div>
          <div className="min-w-0">
            <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">
              <span className="ptc-gradient-text">Path to Care</span>
            </h1>
            <p className="mt-1 text-sm font-medium uppercase tracking-widest text-slate-500">
              Rural healthcare triage decision-support · never diagnoses
            </p>
          </div>
        </div>

        <p className="mt-5 max-w-3xl text-base leading-relaxed text-slate-700">
          Multimodal, agentic decision-support for rural healthcare. The system{" "}
          <strong className="text-slate-900">never diagnoses.</strong> It (1) ranks plausible skin
          conditions as top-3 with confidence, (2) assesses urgency{" "}
          <span className="font-semibold text-rose-600">Red</span> /{" "}
          <span className="font-semibold text-amber-600">Yellow</span> /{" "}
          <span className="font-semibold text-emerald-600">Green</span>, (3) flags red signs, and
          (4) frames the cost-benefit of travelling to the clinic.
        </p>
      </header>

      <InputForm
        onResult={setResult}
        onError={setError}
        onLoading={setLoading}
        loading={loading}
      />

      {error && (
        <div className="mt-4 rounded-xl border border-rose-200/70 bg-rose-50/80 p-4 text-sm text-rose-900 shadow-sm backdrop-blur-sm">
          <span className="mr-1.5 font-semibold">⚠️ Error:</span>{error}
        </div>
      )}

      {result && (
        <div className="mt-10">
          <Tabs
            tabs={[
              { id: "patient", label: "For the patient", icon: "👤", content: <PatientView r={result} /> },
              { id: "clinician", label: "For the clinician", icon: "🩺", content: <ClinicianView r={result} /> },
              { id: "audit", label: "Audit trail", icon: "⚙️", content: <AuditView r={result} /> },
            ]}
          />
        </div>
      )}

      <footer className="mt-16 border-t border-slate-200/70 pt-6 text-center text-xs text-slate-500">
        Built in 24 hrs for the <span className="font-semibold text-slate-700">AMD Developer Hackathon</span>{" "}
        · Inference on AMD MI300X via vLLM ROCm Docker · The clinician diagnoses; the patient decides.
      </footer>
    </main>
  );
}
