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

      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h3 className="text-base font-semibold text-gray-900">What this likely is</h3>
        <p className="mt-2 text-sm leading-relaxed text-gray-700">{r.reasoning}</p>
      </section>

      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h3 className="text-base font-semibold text-gray-900">Should you go to the clinic?</h3>
        <p className="mt-2 text-sm leading-relaxed text-gray-700">{r.patient_framing}</p>
        <p className="mt-3 rounded-md bg-blue-50 p-3 text-xs text-blue-900">{r.village.blurb}</p>
      </section>

      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h3 className="text-base font-semibold text-gray-900">Watch for these signs</h3>
        <p className="mt-1 text-xs text-gray-500">If any of these happen, return immediately.</p>
        <ul className="mt-3 list-disc list-inside text-sm text-gray-800">
          {(r.red_flags_noted.length > 0
            ? r.red_flags_noted
            : ["spreading redness", "high fever", "severe pain", "breathing trouble"]
          ).map((f, i) => <li key={i}>{f}</li>)}
        </ul>
      </section>

      <p className="rounded-md bg-gray-100 p-3 text-xs text-gray-600">{PATIENT_DISCLAIMER}</p>
    </div>
  );
}

function ClinicianView({ r }: { r: TriageResult }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-700">
            Image — top-3 differential
          </h3>
          <ConditionList items={r.image_top3} />
        </div>
        <div className="lg:col-span-2">
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-700">
            Pre-visit SOAP
          </h3>
          <SoapCard soap={r.soap} />
        </div>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-700">
          Red flags — rule-based vs. model
        </h3>
        <RedFlagPanels
          ruleBased={r.cross_check_red_flags}
          modelNoted={r.red_flags_noted}
        />
      </div>

      <div className="rounded-xl border border-gray-200 bg-white p-5">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-700">Triage</h3>
        <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-3">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-gray-500">Urgency</div>
            <div className="mt-1 text-2xl font-semibold uppercase">{r.urgency}</div>
            {r.safety_escalation && (
              <div className="mt-1 text-xs text-amber-700">↑ escalated by safety net</div>
            )}
          </div>
          <div className="md:col-span-2">
            <div className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Reasoning
            </div>
            <p className="mt-1 text-sm text-gray-800">{r.reasoning}</p>
          </div>
        </div>
      </div>

      <p className="rounded-md bg-gray-100 p-3 text-xs text-gray-600">{CLINICIAN_DISCLAIMER}</p>
    </div>
  );
}

function AuditView({ r }: { r: TriageResult }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Wall time" value={`${r.wall_seconds.toFixed(1)} s`} />
        <Stat label="Parse OK" value={r.parse_ok ? "yes" : "no"} />
        <Stat label="Cardinal-rule rewrites" value={String(r.cardinal_rule_rewrites)} />
        <Stat label="Safety escalation" value={r.safety_escalation ? "yes" : "no"} />
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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-gray-200 bg-white p-3">
      <div className="text-xs font-medium uppercase tracking-wide text-gray-500">{label}</div>
      <div className="mt-1 font-mono text-sm text-gray-900">{value}</div>
    </div>
  );
}

export default function Home() {
  const [result, setResult] = useState<TriageResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  return (
    <main className="mx-auto max-w-5xl px-4 pb-12 pt-8">
      <header className="mb-6">
        <div className="flex items-center gap-3">
          <span className="text-3xl">🩺</span>
          <h1 className="text-2xl font-bold text-gray-900">Path to Care</h1>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-gray-600">
          Multimodal, agentic decision-support for rural healthcare. The system{" "}
          <strong>never diagnoses.</strong> It (1) ranks plausible skin conditions, (2) assesses
          urgency Red / Yellow / Green, (3) flags red signs, and (4) frames the cost-benefit of
          travelling to the clinic.
        </p>
        <p className="mt-2 text-xs text-gray-500">
          Built for the AMD Developer Hackathon · Gemma 4 31B-it on MI300X via vLLM ·{" "}
          <a className="text-blue-600 hover:underline" href="https://github.com/SankarSubbayya/path_to_care">GitHub</a>
        </p>
      </header>

      <InputForm
        onResult={setResult}
        onError={setError}
        onLoading={setLoading}
        loading={loading}
      />

      {error && (
        <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-8">
          <Tabs
            tabs={[
              { id: "patient", label: "For the patient", icon: "👤", content: <PatientView r={result} /> },
              { id: "clinician", label: "For the clinician", icon: "🩺", content: <ClinicianView r={result} /> },
              { id: "audit", label: "Audit trail", icon: "⚙️", content: <AuditView r={result} /> },
            ]}
          />
        </div>
      )}
    </main>
  );
}
