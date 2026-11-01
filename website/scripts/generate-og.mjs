import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const root = new URL("../", import.meta.url);
const icon = await readFile(fileURLToPath(new URL("public/icon.png", root)));
const iconBase64 = icon.toString("base64");

const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <radialGradient id="glow-blue" cx="0" cy="0" r="1" gradientTransform="translate(945 90) rotate(135) scale(520)">
      <stop stop-color="#60a5fa" stop-opacity="0.24" />
      <stop offset="1" stop-color="#60a5fa" stop-opacity="0" />
    </radialGradient>
    <radialGradient id="glow-violet" cx="0" cy="0" r="1" gradientTransform="translate(170 560) rotate(-40) scale(430)">
      <stop stop-color="#a78bfa" stop-opacity="0.16" />
      <stop offset="1" stop-color="#a78bfa" stop-opacity="0" />
    </radialGradient>
    <linearGradient id="signal" x1="0" y1="0" x2="1" y2="1">
      <stop stop-color="#60a5fa" />
      <stop offset="0.55" stop-color="#a78bfa" />
      <stop offset="1" stop-color="#38bdf8" />
    </linearGradient>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M40 0H0V40" fill="none" stroke="#94a3b8" stroke-opacity="0.08" stroke-width="1" />
    </pattern>
    <filter id="shadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="24" stdDeviation="28" flood-color="#020617" flood-opacity="0.55" />
    </filter>
    <clipPath id="badge">
      <rect x="620" y="120" width="136" height="136" rx="30" />
    </clipPath>
  </defs>

  <rect width="1200" height="630" fill="#0b0b1a" />
  <rect width="1200" height="630" fill="url(#grid)" />
  <rect width="1200" height="630" fill="url(#glow-blue)" />
  <rect width="1200" height="630" fill="url(#glow-violet)" />

  <text x="80" y="140" fill="#fbbf24" font-family="'SFMono-Regular', 'JetBrains Mono', Menlo, monospace" font-size="18" font-weight="600" letter-spacing="3.5">CLIVER / FIELD NOTES</text>

  <text fill="#ffffff" font-family="'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif" font-size="78" font-weight="600" letter-spacing="-3">
    <tspan x="80" y="244">Field notes</tspan>
    <tspan x="80" y="326">from the agent</tspan>
    <tspan x="80" y="408">layer.</tspan>
  </text>

  <rect x="80" y="454" width="94" height="4" rx="2" fill="url(#signal)" />

  <text x="80" y="508" fill="#cbd5e1" font-family="'SFMono-Regular', 'JetBrains Mono', Menlo, monospace" font-size="19" letter-spacing="1.2">SHORT · TESTED · TWICE A WEEK</text>
  <text x="80" y="556" fill="#64748b" font-family="'SFMono-Regular', 'JetBrains Mono', Menlo, monospace" font-size="16">cliver-project.github.io/CLIver/blog</text>

  <g filter="url(#shadow)">
    <rect x="736" y="198" width="392" height="252" rx="18" fill="#0f172a" stroke="#334155" stroke-width="1.5" />
    <line x1="736" y1="246" x2="1128" y2="246" stroke="#1e293b" stroke-width="1.5" />
    <circle cx="770" cy="222" r="6" fill="#f87171" />
    <circle cx="792" cy="222" r="6" fill="#fbbf24" />
    <circle cx="814" cy="222" r="6" fill="#34d399" />
    <text font-family="'SFMono-Regular', 'JetBrains Mono', Menlo, monospace" font-size="18">
      <tspan x="768" y="292" fill="#fbbf24">$</tspan>
      <tspan fill="#cbd5e1"> cliver run first_agent.py</tspan>
      <tspan x="768" y="334" fill="#64748b">→ agent.chat("...")</tspan>
      <tspan x="768" y="376" fill="#34d399">✓ done in 1.2s</tspan>
    </text>
    <rect x="768" y="392" width="11" height="22" fill="#fbbf24" />
  </g>

  <g filter="url(#shadow)">
    <rect x="620" y="120" width="136" height="136" rx="30" fill="#ffffff" stroke="#e2e8f0" stroke-width="2" />
    <image href="data:image/png;base64,${iconBase64}" x="638" y="138" width="100" height="100" clip-path="url(#badge)" />
  </g>
</svg>
`;

await sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).toFile(
  fileURLToPath(new URL("public/og/blog-default.png", root)),
);

console.log("Wrote public/og/blog-default.png");
