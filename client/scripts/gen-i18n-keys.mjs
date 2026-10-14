// Generates client/src/app/core/i18n/keys.ts from i18n/translations/en.json.
// Leaf values are the dot-path translation keys consumed by the `translate` pipe
// and I18nService.t(). The object mirrors the bundle structure verbatim (keys are
// kept as-is; non-identifier keys are quoted and accessed via bracket notation).
//
// Run from the client/ directory:  node scripts/gen-i18n-keys.mjs
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = path.resolve(import.meta.dirname, '..', '..');
const enPath = path.join(repoRoot, 'i18n/translations/en.json');
const outPath = path.join(repoRoot, 'client/src/app/core/i18n/keys.ts');

const en = JSON.parse(fs.readFileSync(enPath, 'utf8'));
const isIdent = (k) => /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(k);

function emit(obj, prefix, indent) {
  const pad = '  '.repeat(indent);
  const lines = [];
  for (const [k, v] of Object.entries(obj)) {
    const dot = prefix ? `${prefix}.${k}` : k;
    const key = isIdent(k) ? k : JSON.stringify(k);
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      lines.push(`${pad}${key}: {`);
      lines.push(emit(v, dot, indent + 1));
      lines.push(`${pad}},`);
    } else {
      lines.push(`${pad}${key}: ${JSON.stringify(dot)},`);
    }
  }
  return lines.join('\n');
}

const header =
  '// AUTO-GENERATED from i18n/translations/en.json by scripts/gen-i18n-keys.mjs.\n' +
  '// Do not edit by hand -- run `node scripts/gen-i18n-keys.mjs` to regenerate.\n' +
  '// Leaf values are the dot-path translation keys used by the `translate` pipe and I18nService.t().\n' +
  '//\n' +
  '// i18nKeys() returns the same table, for class-field initializers: always write\n' +
  '// `protected readonly I18N = i18nKeys();`, never `= I18N` or any other bare imported binding.\n' +
  "// Vitest's unit-test builder runs specs through Vite's SSR transform, which turns a class\n" +
  '// field whose initializer is a bare imported identifier into a `const X = ns.X` snapshot\n' +
  '// hoisted above the enclosing top-level statement. With the zone.js test polyfill the builder\n' +
  '// lowers async arrows to `__async(this, ...)`; that top-level `this` makes esbuild wrap the\n' +
  '// spec as CommonJS and initialize this module lazily inside `init_keys()`, so the snapshot\n' +
  '// is taken before the table exists and every template binding dereferences `undefined`.\n' +
  '// A call is rewritten to a live namespace access and runs at construction, after init.\n\n';
const body =
  `export const I18N = {\n${emit(en, '', 1)}\n} as const;\n\n` +
  '/** Returns {@link I18N}; use it to initialize class fields -- see the header comment above. */\n' +
  'export function i18nKeys(): typeof I18N {\n  return I18N;\n}\n';

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, header + body);

let leaves = 0;
(function count(o) {
  for (const v of Object.values(o)) {
    if (v && typeof v === 'object' && !Array.isArray(v)) count(v);
    else leaves += 1;
  }
})(en);
console.log(`wrote ${path.relative(repoRoot, outPath)} (${leaves} keys)`);
