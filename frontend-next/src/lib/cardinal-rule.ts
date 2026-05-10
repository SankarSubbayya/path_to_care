// TypeScript port of core/cardinal_rule.py — regex-rewrites diagnostic
// phrases on every model output before patient-facing text leaves the API.

const PATTERNS: Array<[RegExp, string]> = [
  [/\bthe\s+diagnosis\s+is\b/gi, "a possibility is"],
  [/\bdiagnosis\s+is\b/gi, "a possibility is"],
  [/\bdiagnosed\s+with\b/gi, "showing signs of"],
  [/\byou\s+have\b/gi, "signs suggest"],
  [/\bI\s+confirm\b/gi, "appearances suggest"],
  [/\bthis\s+is\s+(definitely|clearly)\b/gi, "signs are consistent with $1"],
  [/\bdefinitely\b/gi, "likely"],
  [/\bcertainly\b/gi, "likely"],
];

export interface RewriteResult {
  text: string;
  rewrites: number;
}

export function enforceCardinalRule(input: string): RewriteResult {
  let text = input;
  let rewrites = 0;
  for (const [pat, repl] of PATTERNS) {
    const next = text.replace(pat, repl);
    if (next !== text) {
      rewrites += 1;
      text = next;
    }
  }
  return { text, rewrites };
}

const RED_FLAG_KEYWORDS = [
  "tetanus", "trismus", "snake", "snakebite", "fang",
  "anaphylax", "throat tight", "lips swelling",
  "necrotizing", "crepitus", "pain disproportionate", "out of proportion",
  "gangrene", "altered mental",
  "spreading erythema", "lymphangitic", "red streaks",
  "rigors", "shivering", "fever 39", "fever 40",
];

export function crossCheckRedFlags(narrative: string, imageDescription?: string): string[] {
  const blob = `${narrative}\n${imageDescription ?? ""}`.toLowerCase();
  return RED_FLAG_KEYWORDS.filter((kw) => blob.includes(kw));
}
