import { useState, useEffect } from "react";
import { useParams, Link, useNavigate } from "react-router";
import { ArrowLeft, Trash2, Save, Loader2, Star, CheckCircle, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useAgents, useConfig, useSaveConfig, useModels } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

interface AgentFormData {
  name: string;
  description: string;
  role: string;
  system_prompt: string;
  model: string;
  isDefault: boolean;
}

function emptyForm(): AgentFormData {
  return { name: "", description: "", role: "", system_prompt: "", model: "", isDefault: false };
}

export default function AgentDetailPage() {
  const { t } = useTranslation();
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const isNew = !name;

  const { data: agentsData, isLoading: agentsLoading } = useAgents();
  const { data: configData } = useConfig();
  const { data: modelsData } = useModels();
  const saveConfig = useSaveConfig();

  const modelList = (modelsData as { models?: string[] })?.models ?? [];
  const config = configData as Record<string, unknown> | undefined;
  const defaultAgent = (config?.default_agent as string) || "";

  const agents = (agentsData ?? []) as Array<Record<string, unknown>>;
  const agent = name ? agents.find((a) => String(a.name) === name) : undefined;

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
        isDefault: defaultAgent === String(agent.name),
      });
    }
  }, [agent, defaultAgent]);

  const set = <K extends keyof AgentFormData>(k: K, v: AgentFormData[K]) =>
    setForm((prev) => ({ ...prev, [k]: v }));

  const handleSave = () => {
    if (!config) return;
    const currentAgents = (config.agents ?? {}) as Record<string, Record<string, unknown>>;
    const agentPayload: Record<string, unknown> = {
      description: form.description || null,
      role: form.role || null,
      model: form.model || null,
    };
    if (form.system_prompt) {
      if (form.role) {
        agentPayload.role = `${form.role}\n\n${form.system_prompt}`.trim();
      } else {
        agentPayload.role = form.system_prompt;
      }
    }
    const updated = { ...currentAgents, [form.name]: agentPayload };
    const updatedConfig = { ...config, agents: updated };
    if (form.isDefault) {
      updatedConfig.default_agent = form.name;
    } else if (defaultAgent === form.name) {
      updatedConfig.default_agent = null;
    }
    saveConfig.mutate(updatedConfig, {
      onSuccess: () => {
        setSaveSuccess(true);
        if (isNew) navigate(`/admin/agents/${encodeURIComponent(form.name)}`);
      },
    });
  };

  const handleDelete = () => {
    if (!config || !name) return;
    const currentAgents = (config.agents ?? {}) as Record<string, Record<string, unknown>>;
    const updated = { ...currentAgents };
    delete updated[name];
    const updatedConfig = { ...config, agents: updated };
    if (defaultAgent === name) updatedConfig.default_agent = null;
    saveConfig.mutate(updatedConfig, { onSuccess: () => navigate("/admin/agents") });
  };

  if (agentsLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;
  if (!isNew && !agent) return <p className="text-muted-foreground">{t("agents.noAgents")}</p>;

  const selectClass = "w-full rounded-md border border-input bg-background px-3 py-2 text-sm";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to="/admin/agents">
          <Button variant="ghost" size="icon"><ArrowLeft className="w-4 h-4" /></Button>
        </Link>
        <h1 className="text-2xl font-bold flex-1">{isNew ? t("agents.createAgent") : name}</h1>
        <Button size="sm" onClick={handleSave} disabled={saveConfig.isPending || !form.name.trim()}>
          {saveConfig.isPending ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Save className="w-4 h-4 mr-1" />}
          {saveConfig.isPending ? t("common.loading") : t("agents.saveAgent")}
        </Button>
        {!isNew && (
          <Button size="sm" variant="destructive" onClick={() => setConfirmDelete(true)}>
            <Trash2 className="w-4 h-4 mr-1" />{t("common.delete")}
          </Button>
        )}
      </div>

      {/* Save feedback */}
      {saveSuccess && !saveConfig.isPending && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-sm">
          <CheckCircle className="w-4 h-4 shrink-0" />{t("config.saved")}
        </div>
      )}
      {saveConfig.isError && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-sm">
          <XCircle className="w-4 h-4 shrink-0" />{t("config.saveError", { error: saveConfig.error?.message ?? "Unknown error" })}
        </div>
      )}

      <ConfirmDialog
        open={confirmDelete}
        title={t("agents.deleteAgent")}
        description={t("agents.deleteAgentDescription", { name: name ?? "" })}
        destructive
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => { setConfirmDelete(false); handleDelete(); }}
      />

      {/* Edit form — always shown */}
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-1">
            <Label>{t("agents.agentName")}</Label>
            <Input value={form.name} onChange={(e) => set("name", e.target.value)} placeholder={t("agents.agentNamePlaceholder")} disabled={!isNew} required />
          </div>

          <div className="space-y-1">
            <Label>{t("agents.description")}</Label>
            <Input value={form.description} onChange={(e) => set("description", e.target.value)} placeholder={t("agents.descriptionPlaceholder")} />
          </div>

          <div className="space-y-1">
            <Label>{t("agents.role")}</Label>
            <textarea className="w-full min-h-[72px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y" value={form.role} onChange={(e) => set("role", e.target.value)} placeholder={t("agents.rolePlaceholder")} rows={3} />
          </div>

          <div className="space-y-1">
            <Label>{t("agents.systemPrompt")}</Label>
            <textarea className="w-full min-h-[140px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y" value={form.system_prompt} onChange={(e) => set("system_prompt", e.target.value)} placeholder={t("agents.systemPromptPlaceholder")} rows={6} />
          </div>

          <div className="space-y-1">
            <Label>{t("agents.model")}</Label>
            <select className={selectClass} value={form.model} onChange={(e) => set("model", e.target.value)}>
              <option value="">{t("agents.noneSelected")}</option>
              {modelList.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          <div className="flex items-center gap-2 pt-1">
            <input type="checkbox" id="isDefault" checked={form.isDefault} onChange={(e) => set("isDefault", e.target.checked)} className="accent-primary" />
            <Label htmlFor="isDefault" className="flex items-center gap-1 cursor-pointer">
              <Star className="w-3.5 h-3.5 text-amber-500" /> {t("agents.isDefault")}
            </Label>
          </div>

        </CardContent>
      </Card>
    </div>
  );
}
