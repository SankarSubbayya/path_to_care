"use client";

// CameraCapture: browser half of the camera_capture MCP.
// Opens the user's camera (rear-facing if available), shows a live preview,
// snaps a frame to a <canvas>, converts to a JPEG File, and passes it up to
// the parent. Falls back gracefully when getUserMedia is unavailable or the
// user denies the permission prompt — the existing file <input> remains the
// path-of-last-resort.

import { useEffect, useRef, useState } from "react";

interface Props {
  onCapture: (file: File, dataUrl: string) => void;
  disabled?: boolean;
}

export function CameraCapture({ onCapture, disabled }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  function stopStream() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  async function start() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera not supported in this browser. Use the file picker below.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      streamRef.current = stream;
      setOpen(true);
      // Wait one tick so the <video> element is mounted before we attach.
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {/* autoplay may block silently */});
        }
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (/permission|denied|notallowed/i.test(msg)) {
        setError("Camera permission denied. Use the file picker below instead.");
      } else {
        setError(`Camera unavailable: ${msg}`);
      }
    }
  }

  function close() {
    stopStream();
    setOpen(false);
  }

  function snap() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const w = video.videoWidth || 1280;
    const h = video.videoHeight || 720;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, w, h);
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError("Snapshot failed (canvas.toBlob returned null).");
          return;
        }
        const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        setPreviewUrl(url);
        onCapture(file, url);
        close();
      },
      "image/jpeg",
      0.9,
    );
  }

  return (
    <div className="space-y-2">
      {!open && (
        <div className="flex items-center gap-3">
          <button
            type="button"
            disabled={disabled}
            onClick={start}
            className="rounded-md border border-blue-600 bg-blue-50 px-3 py-1.5 text-sm font-medium text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Open camera
          </button>
          {previewUrl && (
            <span className="flex items-center gap-2 text-xs text-gray-600">
              <img src={previewUrl} alt="capture preview" className="h-10 w-10 rounded object-cover" />
              captured
            </span>
          )}
          <span className="text-xs text-gray-500">camera_capture MCP — rear camera if available</span>
        </div>
      )}

      {open && (
        <div className="space-y-2 rounded-md border border-gray-300 p-2">
          <video
            ref={videoRef}
            playsInline
            muted
            className="w-full max-h-80 rounded bg-black"
          />
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={snap}
              className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
            >
              Snap
            </button>
            <button
              type="button"
              onClick={close}
              className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {error && <p className="text-xs text-red-600">{error}</p>}

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
