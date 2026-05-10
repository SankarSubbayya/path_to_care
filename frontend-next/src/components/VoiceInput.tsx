"use client";

// VoiceInput: dictate the patient narrative using the browser's Web Speech API.
// Appends final transcripts to whatever is already in the textarea (so the
// typed and dictated paths compose). Falls back gracefully on Firefox / iOS
// where the API is missing — the textarea remains the canonical input.
//
// We keep the recognizer un-continuous and re-start it on each click so the
// mobile lifecycle (Chrome stops after ~30 s of silence) is predictable.

import { useEffect, useRef, useState } from "react";

interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<{
    isFinal: boolean;
    0: { transcript: string };
  }>;
}

interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onerror: ((ev: { error: string }) => void) | null;
  onend: (() => void) | null;
}

type RecognitionCtor = new () => SpeechRecognitionLike;

interface Props {
  onAppend: (text: string) => void;
  disabled?: boolean;
  lang?: string;
}

function getRecognitionCtor(): RecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: RecognitionCtor;
    webkitSpeechRecognition?: RecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function VoiceInput({ onAppend, disabled, lang = "en-US" }: Props) {
  const recRef = useRef<SpeechRecognitionLike | null>(null);
  const [supported, setSupported] = useState<boolean>(true);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const Ctor = getRecognitionCtor();
    setSupported(!!Ctor);
    return () => {
      recRef.current?.stop();
      recRef.current = null;
    };
  }, []);

  function start() {
    setError(null);
    setInterim("");
    const Ctor = getRecognitionCtor();
    if (!Ctor) {
      setSupported(false);
      return;
    }
    const rec = new Ctor();
    rec.lang = lang;
    rec.continuous = false;
    rec.interimResults = true;
    rec.onresult = (ev) => {
      let interimText = "";
      const finals: string[] = [];
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        if (r.isFinal) finals.push(r[0].transcript);
        else interimText += r[0].transcript;
      }
      if (finals.length) {
        const joined = finals.join(" ").trim();
        if (joined) onAppend(joined);
      }
      setInterim(interimText);
    };
    rec.onerror = (ev) => {
      setError(`mic: ${ev.error}`);
    };
    rec.onend = () => {
      setListening(false);
      setInterim("");
    };
    recRef.current = rec;
    try {
      rec.start();
      setListening(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setListening(false);
    }
  }

  function stop() {
    recRef.current?.stop();
  }

  if (!supported) {
    return (
      <p className="text-xs text-gray-500">
        Voice input not supported in this browser — type into the textarea.
      </p>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        disabled={disabled}
        onClick={listening ? stop : start}
        className={
          listening
            ? "rounded-md bg-red-600 px-3 py-1.5 text-sm font-medium text-white shadow hover:bg-red-700 disabled:opacity-50"
            : "rounded-md border border-blue-600 bg-blue-50 px-3 py-1.5 text-sm font-medium text-blue-700 hover:bg-blue-100 disabled:opacity-50"
        }
        aria-pressed={listening}
      >
        {listening ? "Stop dictation" : "Dictate"}
      </button>
      {listening && (
        <span className="text-xs text-gray-600">
          listening… <span className="italic text-gray-500">{interim}</span>
        </span>
      )}
      {!listening && !error && (
        <span className="text-xs text-gray-500">Web Speech API — appends to narrative</span>
      )}
      {error && <span className="text-xs text-red-600">{error}</span>}
    </div>
  );
}
