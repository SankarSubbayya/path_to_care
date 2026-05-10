import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Path to Care — Rural healthcare triage decision support",
  description:
    "Multimodal, agentic decision-support for rural healthcare. Never diagnoses. " +
    "Ranks possibilities, assesses urgency Red/Yellow/Green, frames barriers as cost-benefit.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="ptc-bg min-h-screen text-slate-900 antialiased">
        {/* decorative blurred gradient mesh, fixed to viewport */}
        <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
          <div className="absolute -top-40 -left-32 h-[420px] w-[420px] rounded-full bg-teal-300/30 blur-3xl" />
          <div className="absolute top-20 right-[-120px] h-[420px] w-[420px] rounded-full bg-amber-200/40 blur-3xl" />
          <div className="absolute bottom-[-160px] left-1/3 h-[460px] w-[460px] rounded-full bg-blue-200/40 blur-3xl" />
        </div>
        {children}
      </body>
    </html>
  );
}
