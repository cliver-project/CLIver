import { useState, useEffect } from "react";
import { useParams, Link, useNavigate } from "react-router";
import {
  ArrowLeft, Pencil, Trash2, Save, Loader2, Star,
  CheckCircle, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { MarkdownView } from "@/components/markdown-view";
import { useAgents, useConfig, useSaveConfig, useModels, useSkills } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

interface AgentFormData {
  name: string;
  description: string;
  role: string;
  system_prompt: string;
  model: string;
  skills: string[];
}

function emptyForm(): AgentFormData {
  return { name: "", description: "", role: "", system_prompt: "", model: "", skills: [] };
}

export default function AgentDetailPage() {
  const { t } = useTranslation();
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const isNew = !name;

  const { data: agentsData, isLoading: agentsLoading } = useAgents();
  const { data: configData } = useConfig();
  const { data: modelsData } = useModels();
  const { data: skillsData } = useSkills();
  const saveConfig = useSaveConfig();

  const modelList = (modelsData as { models?: string[] })?.models ?? [];
  const skillList = ((skillsData ?? []) as Array<Record<string, unknown>>).map((s) => String(s.name));

  const agents = (agentsData ?? []) as Array<Record<string, unknown>>;
  const agent = name ? agents.find((a) => String(a.name) === name) : undefined;
  const isDefault = agent?.is_default === true;

  const [editing, setEditing] = useState(isNew);
  const [form, setForm] = useState<AgentFormData>(emptyForm());
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    if (agent) {
      setForm({
        name: String(agent.name ?? ""),
        description: String(agent.description ?? ""),
        role: String(agent.role ?? ""),
        system_prompt: String(agent.system_prompt ?? ""),
        model: String(agent.model ?? ""),
        skills: (agent.skills ?? []) as string[],
      });
    }
  }, [agent]);

  const set = <K extends keyof AgentFormData>(k: K, v: AgentFormData[K]) =>
    setForm((prev) => ({ ...prev, [k]: v }));

  const toggleSkill = (skill: string) => {
    setForm((prev) => ({
      ...prev,
      skills: prev.skills.includes(skill)
        ? prev.skills.filter((s) => s !== skill)
        : [...prev.skills, skill],
    }));
  };

  const handleSave = () => {
    if (!configData) return;
    const config = configData as Record<string, unknown>;
    const currentAgents = (config.agents ?? {}) as Record<string, Record<string, unknown>>;
    const agentPayload: Record<string, unknown> = {
      description: form.description || null,
      role: form.role || null,
      system_prompt: form.system_prompt || null,
      model: form.model || null,
      skills: form.skills.length > 0 ? form.skills : [],
    };
    const updated = { ...currentAgents, [form.name]: agentPayload };
    saveConfig.mutate(
      { ...config, agents: updated },
      {
        onSuccess: () => {
          setSaveSuccess(true);
          if (isNew) {
            navigate(`/admin/agents/${encodeURIComponent(form.name)}`);
          } else {
            setEditing(false);
          }
        },
      },
    );
  };

  const handleDelete = () => {
    if (!configData || !name) return;
    const config = configData as Record<string, unknown>;
    const currentAgents = (config.agents ?? {}) as Record<string, Record<string, unknown>>;
    const updated = { ...currentAgents };
    delete updated[name];
    saveConfig.mutate(
      { ...config, agents: updated },
      { onSuccess: () => navigate("/admin/agents") },
    );
  };

  if (agentsLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;
  if (!isNew && !agent) return <p className="text-muted-foreground">{t("agents.noAgents")}</p>;

  const selectClass = "w-full rounded-md border border-input bg-background px-3 py-2 text-sm";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to="/admin/agents">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-4 h-4" />
          </Button>
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold">{isNew ? t("agents.createAgent") : name}</h1>
          {!isNew && isDefault && (
            <div className="flex items-center gap-2 mt-1">
              <Badge variant="secondary" className="gap-1">
                <Star className="w-3 h-3" />
                {t("agents.defaultBadge")}
              </Badge>
            </div>
          )}
        </div>
        {!isNew && !editing && (
          <>
            <Button size="sm" variant="outline" onClick={() => { setEditing(true); setSaveSuccess(false); }}>
              <Pencil className="w-4 h-4 mr-1" />
              {t("agents.editAgent")}
            </Button>
            <Button size="sm" variant="destructive" onClick={() => setConfirmDelete(true)}>
              <Trash2 className="w-4 h-4 mr-1" />
              {t("common.delete")}
            </Button>
          </>
        )}
        {editing && (
          <Button
            size="sm"
            onClick={handleSave}
            disabled={saveConfig.isPending || !form.name.trim()}
          >
            {saveConfig.isPending ? (
              <Loader2 className="w-4 h-4 mr-1 animate-spin" />
            ) : (
              <Save className="w-4 h-4 mr-1" />
            )}
            {saveConfig.isPending ? t("common.loading") : t("agents.saveAgent")}
          </Button>
        )}
        {editing && !isNew && (
          <Button size="sm" variant="outline" onClick={() => { setEditing(false); if (agent) { setForm({ name: String(agent.name ?? ""), description: String(agent.description ?? ""), role: String(agent.role ?? ""), system_prompt: String(agent.system_prompt ?? ""), model: String(agent.model ?? ""), skills: (agent.skills ?? []) as string[] }); } }}>
            {t("common.cancel")}
          </Button>
        )}
      </div>

      {/* Save feedback */}
      {saveSuccess && !saveConfig.isPending && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-sm">
          <CheckCircle className="w-4 h-4 shrink-0" />
          {t("config.saved")}
        </div>
      )}
      {saveConfig.isError && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-sm">
          <XCircle className="w-4 h-4 shrink-0" />
          {t("config.saveError", { error: saveConfig.error?.message ?? "Unknown error" })}
        </div>
      )}

      {/* Delete confirm */}
      <ConfirmDialog
        open={confirmDelete}
        title={t("agents.deleteAgent")}
        description={t("agents.deleteAgentDescription", { name: name ?? "" })}
        destructive
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => { setConfirmDelete(false); handleDelete(); }}
      />

      {/* View mode */}
      {!editing && agent && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t("agents.title")}</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 text-sm">
              {agent.description && (
                <>
                  <dt className="text-muted-foreground">{t("agents.description")}</dt>
                  <dd>{String(agent.description)}</dd>
                </>
              )}
              {agent.role && (
                <>
                  <dt className="text-muted-foreground">{t("agents.role")}</dt>
                  <dd>{String(agent.role)}</dd>
                </>
              )}
              {agent.system_prompt && (
                <>
                  <dt className="text-muted-foreground">{t("agents.systemPrompt")}</dt>
                  <dd><MarkdownView content={String(agent.system_prompt)} /></dd>
                </>
              )}
              {agent.model && (
                <>
                  <dt className="text-muted-foreground">{t("agents.model")}</dt>
                  <dd><Badge variant="secondary">{String(agent.model)}</Badge></dd>
                </>
              )}
              <>
                <dt className="text-muted-foreground">{t("agents.skills")}</dt>
                <dd className="flex flex-wrap gap-1">
                  {((agent.skills ?? []) as string[]).length > 0
                    ? ((agent.skills ?? []) as string[]).map((s) => (
                        <Badge key={s} variant="outline">{s}</Badge>
                      ))
                    : <span className="text-muted-foreground">—</span>}
                </dd>
              </>
            </dl>
          </CardContent>
        </Card>
      )}

      {/* Edit mode */}
      {editing && (
        <Card>
          <CardContent className="pt-6 space-y-4">
            <div className="space-y-1">
              <Label>{t("agents.agentName")}</Label>
              <Input
                value={form.name}
                onChange={(e) => set("name", e.target.value)}
                placeholder={t("agents.agentNamePlaceholder")}
                disabled={!isNew}
                required
              />
            </div>

            <div className="space-y-1">
              <Label>{t("agents.description")}</Label>
              <Input
                value={form.description}
                onChange={(e) => set("description", e.target.value)}
                placeholder={t("agents.descriptionPlaceholder")}
              />
            </div>

            <div className="space-y-1">
              <Label>{t("agents.role")}</Label>
              <textarea
                className="w-full min-h-[72px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y"
                value={form.role}
                onChange={(e) => set("role", e.target.value)}
                placeholder={t("agents.rolePlaceholder")}
                rows={3}
              />
            </div>

            <div className="space-y-1">
              <Label>{t("agents.systemPrompt")}</Label>
              <textarea
                className="w-full min-h-[140px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y"
                value={form.system_prompt}
                onChange={(e) => set("system_prompt", e.target.value)}
                placeholder={t("agents.systemPromptPlaceholder")}
                rows={6}
              />
            </div>

            <div className="space-y-1">
              <Label>{t("agents.model")}</Label>
              <select
                className={selectClass}
                value={form.model}
                onChange={(e) => set("model", e.target.value)}
              >
                <option value="">{t("agents.noneSelected")}</option>
                {modelList.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>

            <div className="space-y-1">
              <Label>{t("agents.skills")}</Label>
              <div className="flex flex-wrap gap-2">
                {skillList.length === 0 ? (
                  <span className="text-sm text-muted-foreground">—</span>
                ) : (
                  skillList.map((s) => (
                    <label
                      key={s}
                      className="flex items-center gap-1.5 cursor-pointer text-sm border rounded-md px-2 py-1 hover:bg-accent transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={form.skills.includes(s)}
                        onChange={() => toggleSkill(s)}
                        className="accent-primary"
                      />
                      {s}
                    </label>
                  ))
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
