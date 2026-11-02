#!/usr/bin/env node
/**
 * Validate blog snippet includes and the production build.
 *
 * 1. Every `file=`/`region=` include resolves and extracts, including drafts
 *    (drafts do not render in the production build, so the remark plugin never
 *    sees them there).
 * 2. Every published post includes at least one sample region.
 * 3. No built HTML page leaks the `file=`/`region=` directives.
 *
 * Run with `--sources-only` from `prebuild` (no dist exists yet); the full run
 * after `npm run build` also scans the built HTML.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadRegion } from "../src/plugins/remark-code-snippets.mjs";

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const blogDir = path.join(repoRoot, "website", "src", "content", "blog");
const distDir = path.join(repoRoot, "website", "dist", "blog");
// `npm run build` runs this script via `prebuild`, before a fresh dist exists.
const sourcesOnly = process.argv.includes("--sources-only");
const DIRECTIVE = /\b([A-Za-z][\w-]*)=(?:"([^"]*)"|'([^']*)'|(\S+))/g;
const FENCE = /^```[^\n]*$/gm;

function walk(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(full) : [full];
  });
}

function frontmatter(text) {
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)/);
  return match ? match[1] : "";
}

function isDraft(frontmatterText) {
  return /^draft:\s*true\s*$/m.test(frontmatterText);
}

function includesIn(text) {
  const body = text.replace(/^---\r?\n[\s\S]*?\r?\n---\s*(?:\r?\n|$)/, "");
  const includes = [];
  for (const fence of body.matchAll(FENCE)) {
    const directives = new Map();
    for (const match of fence[0].matchAll(DIRECTIVE)) {
      directives.set(match[1], match[2] ?? match[3] ?? match[4]);
    }
    if (directives.has("file") || directives.has("region")) includes.push(directives);
  }
  return includes;
}

const failures = [];
const posts = walk(blogDir).filter((file) => /\.mdx?$/.test(file));
let includeCount = 0;

for (const post of posts) {
  const relativePost = path.relative(repoRoot, post);
  const text = fs.readFileSync(post, "utf8");
  const draft = isDraft(frontmatter(text));
  const includes = includesIn(text);

  if (!draft && includes.length === 0) {
    failures.push(`${relativePost}: published post has no file=/region= snippet`);
  }

  for (const directives of includes) {
    includeCount += 1;
    const sampleRef = directives.get("file");
    const region = directives.get("region");
    if (!sampleRef) {
      failures.push(`${relativePost}: region= without file=`);
      continue;
    }
    if (!region) {
      failures.push(`${relativePost}: file= without region= (${sampleRef})`);
      continue;
    }
    try {
      loadRegion(repoRoot, { blogPath: relativePost, sampleRef, region });
    } catch (error) {
      failures.push(error.message);
    }
  }
}

let htmlChecked = 0;
if (!sourcesOnly && fs.existsSync(distDir)) {
  for (const html of walk(distDir).filter((file) => file.endsWith(".html"))) {
    htmlChecked += 1;
    const text = fs.readFileSync(html, "utf8");
    if (/\bfile=("|')samples\//.test(text) || /\bregion=("|')/.test(text)) {
      failures.push(`${path.relative(repoRoot, html)}: built page still contains snippet directives`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log(
  `Blog snippets OK: ${posts.length} posts, ${includeCount} includes, ${htmlChecked} built pages checked`,
);
