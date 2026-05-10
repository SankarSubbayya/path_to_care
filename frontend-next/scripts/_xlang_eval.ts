import { enforceCardinalRule } from '../src/lib/cardinal-rule.ts';
const probes: string[] = JSON.parse(process.argv[2]);
const out = probes.map((p) => enforceCardinalRule(p).text);
process.stdout.write(JSON.stringify(out));
