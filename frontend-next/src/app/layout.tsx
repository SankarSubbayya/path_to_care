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
      <body className="bg-gray-50 text-gray-900 antialiased">{children}</body>
    </html>
  );
}
