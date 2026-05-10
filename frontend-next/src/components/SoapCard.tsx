import type { SoapFields } from "@/lib/types";

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="border-t border-gray-200 first:border-t-0 first:pt-0 pt-3 mt-3 first:mt-0">
      <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">{label}</div>
      <div className="mt-1 text-sm text-gray-900">{children}</div>
    </div>
  );
}

function ListOrNone({ items, italic = false }: { items?: string[]; italic?: boolean }) {
  if (!items || items.length === 0)
    return <span className="italic text-gray-400">none stated</span>;
  return (
    <ul className={`list-disc list-inside ${italic ? "italic" : ""}`}>
      {items.map((s, i) => (
        <li key={i}>{s}</li>
      ))}
    </ul>
  );
}

export function SoapCard({ soap }: { soap: SoapFields }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <Section label="Chief complaint">{soap.chief_complaint || <span className="italic text-gray-400">none stated</span>}</Section>
      <Section label="HPI (history of present illness)">{soap.hpi || <span className="italic text-gray-400">none stated</span>}</Section>
      <Section label="Duration">{soap.duration || <span className="italic text-gray-400">unknown</span>}</Section>
      <Section label="Associated symptoms"><ListOrNone items={soap.associated_symptoms} /></Section>
      <Section label="Past medical history"><ListOrNone items={soap.past_medical_history} /></Section>
      <Section label="Medications"><ListOrNone items={soap.medications} /></Section>
      <Section label="Exam findings"><ListOrNone items={soap.exam_findings} /></Section>
      <Section label="Red flags from SOAP"><ListOrNone items={soap.red_flags} /></Section>
      <Section label="Patient concerns / barriers"><ListOrNone items={soap.patient_concerns} /></Section>
    </div>
  );
}
