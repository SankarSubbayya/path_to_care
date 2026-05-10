"use client";

import { useState } from "react";
import type { TriageResult } from "@/lib/types";
import { CameraCapture } from "./CameraCapture";
import { VoiceInput } from "./VoiceInput";

const CANNED_NARRATIVE =
  "I cut my foot on a rusty nail two days back when I was working in the field. " +
  "Now my whole foot is swollen and red, the redness is going up my leg. " +
  "I have fever since yesterday night, body shivering. Cannot keep weight on the foot.";

const CANNED_IMAGE_DESCRIPTION =
  "Lower leg with poorly demarcated erythema extending proximally above the wound; " +
  "warmth and edema; small puncture wound visible on plantar foot.";

interface Props {
  onResult: (r: TriageResult) => void;
  onError: (e: string) => void;
  onLoading: (b: boolean) => void;
  loading: boolean;
}

export function InputForm({ onResult, onError, onLoading, loading }: Props) {
  const [narrative, setNarrative] = useState(CANNED_NARRATIVE);
  const [imageDescription, setImageDescription] = useState(CANNED_IMAGE_DESCRIPTION);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [imageSource, setImageSource] = useState<"none" | "camera" | "upload">("none");

  function appendDictation(text: string) {
    setNarrative((prev) => (prev ? `${prev.replace(/\s+$/, "")} ${text}` : text));
  }

  function handleCameraCapture(file: File, dataUrl: string) {
    setImageFile(file);
    setImagePreviewUrl(dataUrl);
    setImageSource("camera");
  }

  function handleFileUpload(file: File | null) {
    setImageFile(file);
    setImagePreviewUrl(file ? URL.createObjectURL(file) : null);
    setImageSource(file ? "upload" : "none");
  }

  async function handleSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    onLoading(true);
    onError("");
    try {
      const fd = new FormData();
      fd.append("narrative", narrative);
      fd.append("image_description", imageDescription);
      if (imageFile) fd.append("image", imageFile);
      if (imageSource !== "none") fd.append("image_source", imageSource);

      const resp = await fetch("/api/triage", { method: "POST", body: fd });
      if (!resp.ok) {
        const j = (await resp.json().catch(() => ({}))) as { error?: string };
        onError(j.error ?? `HTTP ${resp.status}`);
        return;
      }
      const r = (await resp.json()) as TriageResult;
      onResult(r);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      onLoading(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="ptc-card relative space-y-5 overflow-hidden rounded-2xl p-6"
    >
      {/* decorative accent strip */}
      <div aria-hidden className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-teal-500 via-cyan-500 to-amber-500" />

      <div className="flex items-center gap-2">
        <span className="text-lg">📝</span>
        <h2 className="text-base font-semibold text-slate-900">New triage request</h2>
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-700" htmlFor="narrative">
          Patient narrative (typed or dictated)
        </label>
        <textarea
          id="narrative"
          rows={5}
          value={narrative}
          onChange={(e) => setNarrative(e.target.value)}
          className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          placeholder="Describe the symptoms in the patient's own words..."
        />
        <div className="mt-2">
          <VoiceInput onAppend={appendDictation} disabled={loading} />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">
          Photo (optional — uses image_description if omitted)
        </label>
        <div className="mt-1 space-y-2">
          <CameraCapture onCapture={handleCameraCapture} disabled={loading} />
          <input
            id="image"
            type="file"
            accept="image/*"
            onChange={(e) => handleFileUpload(e.target.files?.[0] ?? null)}
            className="block w-full text-sm text-gray-700 file:mr-3 file:rounded-md file:border-0 file:bg-blue-50 file:px-3 file:py-2 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100"
          />
          {imagePreviewUrl && (
            <div className="flex items-center gap-3 rounded-md border border-gray-200 bg-gray-50 p-2">
              <img src={imagePreviewUrl} alt="selected" className="h-16 w-16 rounded object-cover" />
              <div className="text-xs text-gray-600">
                <div>source: <span className="font-medium">{imageSource}</span></div>
                {imageFile && (
                  <div>
                    {imageFile.name} · {(imageFile.size / 1024).toFixed(0)} KB
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700" htmlFor="image_description">
          What the image shows (text proxy if no photo attached)
        </label>
        <textarea
          id="image_description"
          rows={3}
          value={imageDescription}
          onChange={(e) => setImageDescription(e.target.value)}
          className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        />
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500">
          Inference runs on an AMD MI300X GPU via vLLM. ~1–3 s per request when warm.
        </p>
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-blue-600 px-5 py-2 text-sm font-medium text-white shadow hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Running triage..." : "Run triage"}
        </button>
      </div>
    </form>
  );
}
