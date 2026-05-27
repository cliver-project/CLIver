import { useState, useEffect } from "react";
import { useParams, Link, useNavigate } from "react-router";
import { ArrowLeft, Trash2, Save, Loader2, Star, CheckCircle, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useAgent, useModels, useUpdateAgent, useCreateAgent, useSetDefaultAgent, useDeleteAgent, type AgentInfo } from "@/hooks/use-api";
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
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isNew = !id;

  const { data: agent, isLoading: agentLoading } = useAgent(id);
  const { data: models } = useModels();
  const updateAgent = useUpdateAgent(id ?? "");
  const createAgent = useCreateAgent();
  const setDefaultAgent = useSetDefaultAgent();
  const deleteAgent = useDeleteAgent();

  const [form, setForm] = useState<AgentFormData>(emptyForm());
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    if (agent) {
      setForm({
        name: agent.name ?? "",
        description: agent.description ?? "",
        role: agent.role ?? "",
        system_prompt: "",
        model: agent.model ?? "",
        isDefault: agent.is_default === 1,
      });
    }
  }, [agent]);

  const set = <K extends keyof AgentFormData>(k: K, v: AgentFormData[K]) =>
    setForm((prev) => ({ ...prev, [k]: v }));

  const handleSave = async () => {
    const agentPayload: Partial<AgentInfo> = {
      name: form.name,
      description: form.description || null,
      model: form.model || null,
    };
    let combinedRole = form.role;
    if (form.system_prompt) {
      combinedRole = form.role
        ? `${form.role}\n\n${form.system_prompt}`.trim()
        : form.system_prompt;
    }
    agentPayload.role = combinedRole || null;

    try {
      if (isNew) {
        const created = await createAgent.mutateAsync(agentPayload);
        if (form.isDefault) {
          await setDefaultAgent.mutateAsync(created.id);
        }
        navigate(`/admin/agents/${created.id}`);
      } else {
        await updateAgent.mutateAsync(agentPayload);
        if (form.isDefault && agent && agent.is_default !== 1) {
          await setDefaultAgent.mutateAsync(id!);
        }
        setSaveSuccess(true);
      }
    } catch {
      // error handled by react-query
    }
  };

  const handleDelete = () => {
    if (!id) return;
    deleteAgent.mutate(id, { onSuccess: () => navigate("/admin/agents") });
  };

  if (agentLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;
  if (!isNew && !agent) return <p className="text-muted-foreground">{t("agents.noAgents")}</p>;

  const selectClass = "w-full rounded-md border border-input bg-background px-3 py-2 text-sm";

  const savePending = updateAgent.isPending || createAgent.isPending;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to="/admin/agents">
          <Button variant="ghost" size="icon"><ArrowLeft className="w-4 h-4" /></Button>
        </Link>
        <h1 className="text-2xl font-bold flex-1">{isNew ? t("agents.createAgent") : agent?.name}</h1>
        <Button size="sm" onClick={handleSave} disabled={savePending || !form.name.trim()}>
          {savePending ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Save className="w-4 h-4 mr-1" />}
          {savePending ? t("common.loading") : t("agents.saveAgent")}
        </Button>
        {!isNew && (
          <Button size="sm" variant="destructive" onClick={() => setConfirmDelete(true)}>
            <Trash2 className="w-4 h-4 mr-1" />{t("common.delete")}
          </Button>
        )}
      </div>

      {/* Save feedback */}
      {saveSuccess && !savePending && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-sm">
          <CheckCircle className="w-4 h-4 shrink-0" />{t("config.saved")}
        </div>
      )}
      {(updateAgent.isError || createAgent.isError) && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-sm">
          <XCircle className="w-4 h-4 shrink-0" />{t("config.saveError", { error: (updateAgent.error ?? createAgent.error)?.message ?? "Unknown error" })}
        </div>
      )}

      <ConfirmDialog
        open={confirmDelete}
        title={t("agents.deleteAgent")}
        description={t("agents.deleteAgentDescription", { name: agent?.name ?? "" })}
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
              {(models ?? []).map((m) => <option key={m.id} value={m.name}>{m.name}</option>)}
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
