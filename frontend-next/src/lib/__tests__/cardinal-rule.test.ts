// Mirror of tests/test_cardinal_rule.py. Both files MUST stay aligned —
// the Python and TypeScript rewriters target the same patterns and must
// produce identical output on the same input. The cross-language
// equivalence script (scripts/test_cardinal_rule_cross_lang.py) asserts
// that beyond what either unit test can.

import { describe, expect, it } from "vitest";
import {
  enforceCardinalRule,
  crossCheckRedFlags,
} from "@/lib/cardinal-rule";

describe("enforceCardinalRule — pattern coverage", () => {
  const cases: Array<[string, string]> = [
    ["you have cellulitis",            "signs suggest"],
    ["You have a fever",                "signs suggest"],
    ["YOU   HAVE  high fever",          "signs suggest"],
    ["the diagnosis is sepsis",         "a possibility is"],
    ["Diagnosis is contact dermatitis", "a possibility is"],
    ["diagnosed with impetigo",         "showing signs of"],
    ["I confirm the rash",              "appearances suggest"],
    ["This is definitely an infection", "consistent with"],
    ["This is clearly cellulitis",      "consistent with"],
  ];
  for (const [input, expected] of cases) {
    it(`rewrites: ${JSON.stringify(input)} → contains ${JSON.stringify(expected)}`, () => {
      const out = enforceCardinalRule(input);
      expect(out.text.toLowerCase()).toContain(expected);
      expect(out.rewrites).toBeGreaterThan(0);
    });
  }
});

describe("enforceCardinalRule — soft cleanups", () => {
  it("collapses 'definitely' → 'likely'", () => {
    expect(enforceCardinalRule("This is definitely cellulitis").text.toLowerCase())
      .toContain("likely");
  });
  it("collapses 'certainly' → 'likely'", () => {
    expect(enforceCardinalRule("certainly an infection").text.toLowerCase())
      .toContain("likely");
  });
});

describe("enforceCardinalRule — clean text untouched", () => {
  it("returns identical text + 0 rewrites for already-clean input", () => {
    const input = "Signs suggest cellulitis. The patient should be evaluated within 24 hours.";
    const out = enforceCardinalRule(input);
    expect(out.text).toBe(input);
    expect(out.rewrites).toBe(0);
  });

  it("is idempotent on cleaned output", () => {
    const once = enforceCardinalRule("you have cellulitis");
    const twice = enforceCardinalRule(once.text);
    expect(twice.text).toBe(once.text);
    expect(twice.rewrites).toBe(0);
  });
});

describe("enforceCardinalRule — Y09 case (live during eval)", () => {
  it("rewrites the actual phrase that fired during eval", () => {
    const before =
      "These spots are spreading and you have a fever, so it is important " +
      "to see a nurse in the next day or two.";
    const out = enforceCardinalRule(before);
    expect(out.text).not.toContain("you have a fever");
    expect(out.text.toLowerCase()).toContain("signs suggest a fever");
  });
});

describe("crossCheckRedFlags", () => {
  it.each([
    ["I have fever 39 and shivering", "redness extending up leg", 2],
    ["snake bit me, fang marks visible", "swelling rapid", 1],
    ["just a small scratch healing well", "minor abrasion no signs", 0],
    ["trismus and rigors", "wound discoloration", 2],
  ] as Array<[string, string, number]>)(
    "narrative=%s img=%s expects ≥%i flags",
    (narrative, img, atLeast) => {
      expect(crossCheckRedFlags(narrative, img).length).toBeGreaterThanOrEqual(atLeast);
    }
  );

  it("returns no flags for benign text", () => {
    expect(
      crossCheckRedFlags(
        "Mild dry skin patches that have been there for years.",
        "Localized scaling, no inflammation."
      )
    ).toEqual([]);
  });
});
