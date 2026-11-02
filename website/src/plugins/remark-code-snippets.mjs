/**
 * Remark plugin: replace a fenced code block with a marked region from a file
 * in the repository.
 *
 * Usage in Markdown/MDX:
 *
 *   ```python title="quickstart.py" file="samples/blog/quickstart/quickstart.py" region="agent"
 *   ```
 *
 * The plugin reads the sample from disk, substitutes the region's lines, keeps
 * `title` (and any other Expressive Code metadata), and drops `file`, `region`,
 * and `lang` from the rendered meta. An invalid include throws, so the site
 * build fails instead of shipping a stale or empty snippet.
 *
 * Region markers are flat and language-agnostic:
 *
 *   # region: name            // region: name            -- region: name
 *   ... excerpt ...
 *   # endregion: name         // endregion: name         -- endregion: name
 */
import fs from "node:fs";
import path from "node:path";

const REGION_START = /^\s*(?:#|\/\/|--)\s*region:\s*([A-Za-z0-9_.-]+)\s*$/;
const REGION_END = /^\s*(?:#|\/\/|--)\s*endregion:\s*([A-Za-z0-9_.-]+)\s*$/;
const DIRECTIVE = /\b([A-Za-z][\w-]*)=(?:"([^"]*)"|'([^']*)'|(\S+))/g;

function parseDirectives(meta) {
  const directives = new Map();
  if (!meta) return directives;
  for (const match of meta.matchAll(DIRECTIVE)) {
    directives.set(match[1], match[2] ?? match[3] ?? match[4]);
  }
  return directives;
}

function stripDirectives(meta, names) {
  if (!meta) return null;
  let cleaned = meta;
  for (const name of names) {
    cleaned = cleaned.replace(new RegExp(`(^|\\s)${name}=(?:"[^"]*"|'[^']*'|\\S+)`, "g"), "$1");
  }
  return cleaned.replace(/\s+/g, " ").trim() || null;
}

function includeError({ blogPath, samplePath, region, available = [], detail }) {
  const lines = [`Blog snippet error in ${blogPath}:`, `  sample: ${samplePath}`];
  if (region) lines.push(`  region: ${region}`);
  if (detail) lines.push(`  ${detail}`);
  if (available.length > 0) lines.push(`  available regions: ${available.join(", ")}`);
  return new Error(lines.join("\n"));
}

function resolveSamplePath(repoRoot, sampleRef, blogPath) {
  if (path.isAbsolute(sampleRef)) {
    throw includeError({
      blogPath,
      samplePath: sampleRef,
      detail: "sample paths must be repository-relative",
    });
  }
  const resolved = path.resolve(repoRoot, sampleRef);
  const relative = path.relative(repoRoot, resolved);
  if (relative === "" || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw includeError({
      blogPath,
      samplePath: sampleRef,
      detail: "sample path escapes the repository",
    });
  }
  return resolved;
}

/**
 * Extract one region from `content`, without its marker lines and without
 * surrounding blank lines. Exported so the blog check script validates
 * includes in drafts, which never reach the production build.
 */
export function extractRegion(content, { blogPath, samplePath, region }) {
  const lines = content.split(/\r?\n/);
  const starts = new Map();
  const regions = new Map();
  let open = null;

  lines.forEach((line, index) => {
    const start = line.match(REGION_START);
    if (start) {
      const name = start[1];
      if (open) {
        throw includeError({
          blogPath,
          samplePath,
          region,
          detail: `region "${name}" starts inside region "${open.name}" (nesting is not supported)`,
        });
      }
      if (starts.has(name)) {
        throw includeError({
          blogPath,
          samplePath,
          region,
          detail: `region "${name}" is declared more than once`,
        });
      }
      starts.set(name, index);
      open = { name, start: index };
      return;
    }

    const end = line.match(REGION_END);
    if (!end) return;
    const name = end[1];
    if (!open) {
      throw includeError({
        blogPath,
        samplePath,
        region,
        detail: `endregion: ${name} has no matching region start`,
      });
    }
    if (open.name !== name) {
      throw includeError({
        blogPath,
        samplePath,
        region,
        detail: `region "${open.name}" is closed by endregion: ${name}`,
      });
    }
    regions.set(name, { start: open.start, end: index });
    open = null;
  });

  if (open) {
    throw includeError({
      blogPath,
      samplePath,
      region,
      detail: `region "${open.name}" has no endregion`,
    });
  }

  const found = regions.get(region);
  if (!found) {
    throw includeError({ blogPath, samplePath, region, available: [...starts.keys()] });
  }

  const body = lines.slice(found.start + 1, found.end);
  while (body.length > 0 && body[0].trim() === "") body.shift();
  while (body.length > 0 && body[body.length - 1].trim() === "") body.pop();
  if (body.length === 0) {
    throw includeError({
      blogPath,
      samplePath,
      region,
      available: [...regions.keys()],
      detail: "the region is empty",
    });
  }
  return body.join("\n");
}

/**
 * Resolve, read, and extract one include. Shared by the remark plugin and the
 * blog check script so drafts (which never reach the production build) and
 * published posts report identical errors.
 */
export function loadRegion(repoRoot, { blogPath, sampleRef, region }) {
  const resolved = resolveSamplePath(repoRoot, sampleRef, blogPath);
  let content;
  try {
    content = fs.readFileSync(resolved, "utf8");
  } catch (error) {
    throw includeError({
      blogPath,
      samplePath: sampleRef,
      region,
      detail: `cannot read sample file (${error.code || error.message})`,
    });
  }
  return extractRegion(content, { blogPath, samplePath: sampleRef, region });
}

/**
 * Remark plugin factory. `repoRoot` is the repository root that `file=` paths
 * are resolved against; `astro.config.mjs` passes it explicitly.
 */
export function remarkCodeSnippets({ repoRoot } = {}) {
  if (!repoRoot) throw new Error("remarkCodeSnippets requires a repoRoot option");
  const root = path.resolve(repoRoot);

  return (tree, file) => {
    const sourcePath = file.path || file.history?.[file.history.length - 1] || "";
    const blogPath = sourcePath ? path.relative(root, sourcePath) || sourcePath : "unknown blog file";

    const visit = (node) => {
      if (!node || typeof node !== "object") return;
      if (node.type === "code") transform(node);
      if (Array.isArray(node.children)) node.children.forEach(visit);
    };

    const transform = (node) => {
      const directives = parseDirectives(node.meta);
      if (!directives.has("file")) return;

      const sampleRef = directives.get("file");
      const region = directives.get("region");
      if (!region) {
        throw includeError({ blogPath, samplePath: sampleRef, detail: "file= requires region=" });
      }

      node.value = loadRegion(root, { blogPath, sampleRef, region });
      const lang = directives.get("lang");
      if (lang) node.lang = lang;
      node.meta = stripDirectives(node.meta, ["file", "region", "lang"]);
    };

    visit(tree);
  };
}

export default remarkCodeSnippets;
