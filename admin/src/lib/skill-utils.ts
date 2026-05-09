export interface SkillForm {
  name: string;
  description: string;
  allowedTools: string[];
}

const VALID_NAME_RE = /^[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9])){0,63}$/;

export function validateSkillName(name: string): string | null {
  if (!name) return "Name is required";
  if (name.length > 64) return `Name exceeds 64 characters (${name.length})`;
  if (!VALID_NAME_RE.test(name)) {
    const reasons: string[] = [];
    if (name !== name.toLowerCase()) reasons.push("must be lowercase");
    if (name.startsWith("-") || name.endsWith("-"))
      reasons.push("must not start or end with hyphen");
    if (name.includes("--")) reasons.push("no consecutive hyphens");
    if (/[^a-z0-9-]/.test(name))
      reasons.push("only lowercase letters, digits, and hyphens");
    return reasons.length > 0 ? reasons.join("; ") : "Invalid name format";
  }
  return null;
}

export function emptySkillForm(): SkillForm {
  return { name: "", description: "", allowedTools: [] };
}

export function parseSkillMd(raw: string): { form: SkillForm; body: string } {
  const defaultForm = emptySkillForm();
  if (!raw.startsWith("---")) return { form: defaultForm, body: raw };

  const endIdx = raw.indexOf("---", 3);
  if (endIdx === -1) return { form: defaultForm, body: raw };

  const frontmatter = raw.substring(3, endIdx).trim();
  const body = raw.substring(endIdx + 3).replace(/^\n/, "");

  const form = { ...defaultForm };
  for (const line of frontmatter.split("\n")) {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;
    const key = line.substring(0, colonIdx).trim();
    const value = line.substring(colonIdx + 1).trim();
    switch (key) {
      case "name":
        form.name = value;
        break;
      case "description":
        form.description = value;
        break;
      case "allowed-tools":
        form.allowedTools = value.split(/\s+/).filter(Boolean);
        break;
    }
  }
  return { form, body };
}

export function buildSkillMd(form: SkillForm, body: string): string {
  let frontmatter = `---\nname: ${form.name}\ndescription: ${form.description}\n`;
  if (form.allowedTools.length > 0)
    frontmatter += `allowed-tools: ${form.allowedTools.join(" ")}\n`;
  frontmatter += "---\n";
  return frontmatter + body;
}
