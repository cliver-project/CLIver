import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { useModels, useSkills, useWorkflows } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

export interface TaskFormData {
  name: string;
  prompt: string;
  description: string;
  scheduleType: "manual" | "one-shot" | "cron";
  schedule: string;
  run_at: string;
  model: string;
  skill: string;
  workflow: string;
}

export function taskFormDataFromTask(data: Record<string, unknown>): TaskFormData {
  let scheduleType: TaskFormData["scheduleType"] = "manual";
  if (data.run_at) scheduleType = "one-shot";
  else if (data.schedule) scheduleType = "cron";

  const skills = data.skills as string[] | null;

  return {
    name: String(data.name ?? ""),
    prompt: String(data.prompt ?? ""),
    description: String(data.description ?? ""),
    scheduleType,
    schedule: String(data.schedule ?? ""),
    run_at: String(data.run_at ?? ""),
    model: String(data.model ?? ""),
    skill: skills?.[0] ?? "",
    workflow: String(data.workflow ?? ""),
  };
}

export function taskFormDataToPayload(form: TaskFormData): Record<string, unknown> {
  return {
    name: form.name,
    prompt: form.prompt,
    description: form.description || null,
    model: form.model || null,
    skills: form.skill ? [form.skill] : null,
    workflow: form.workflow || null,
    schedule: form.scheduleType === "cron" ? form.schedule || null : null,
    run_at: form.scheduleType === "one-shot" ? form.run_at || null : null,
  };
}

interface TaskFormProps {
  mode: "create" | "edit";
  initialData?: TaskFormData;
  onSubmit: (data: Record<string, unknown>) => void;
  isPending: boolean;
  error?: string;
}

export default function TaskForm({ mode, initialData, onSubmit, isPending, error }: TaskFormProps) {
  const { t } = useTranslation();
  const { data: modelsData } = useModels();
  const { data: skillsData } = useSkills();
  const { data: workflowsData } = useWorkflows();

  const modelList = (modelsData as { models?: string[] })?.models ?? [];
  const skillList = ((skillsData ?? []) as Array<Record<string, unknown>>).map((s) => String(s.name));
  const workflowList = ((workflowsData ?? []) as Array<Record<string, unknown>>).map((w) => String(w.name));

  const [form, setForm] = useState<TaskFormData>(
    initialData ?? {
      name: "", prompt: "", description: "",
      scheduleType: "manual", schedule: "", run_at: "",
      model: "", skill: "", workflow: "",
    },
  );

  const set = <K extends keyof TaskFormData>(k: K, v: TaskFormData[K]) =>
    setForm((prev) => ({ ...prev, [k]: v }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(taskFormDataToPayload(form));
  };

  const selectClass = "w-full rounded-md border border-input bg-background px-3 py-2 text-sm";

  return (
    <form onSubmit={handleSubmit}>
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-1">
            <Label>{t("tasks.taskName")}</Label>
            <Input value={form.name} onChange={(e) => set("name", e.target.value)}
              placeholder={t("tasks.taskNamePlaceholder")} disabled={mode === "edit"} required />
          </div>

          <div className="space-y-1">
            <Label>{t("tasks.taskPrompt")}</Label>
            <textarea
              className="w-full min-h-[120px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y"
              value={form.prompt} onChange={(e) => set("prompt", e.target.value)}
              placeholder={t("tasks.taskPromptPlaceholder")} required />
          </div>

          <div className="space-y-1">
            <Label>{t("tasks.taskDescription")}</Label>
            <Input value={form.description} onChange={(e) => set("description", e.target.value)}
              placeholder={t("tasks.taskDescriptionPlaceholder")} />
          </div>

          <div className="space-y-2">
            <Label>{t("tasks.scheduleType")}</Label>
            <div className="flex gap-4">
              {(["manual", "one-shot", "cron"] as const).map((st) => (
                <label key={st} className="flex items-center gap-2 cursor-pointer text-sm">
                  <input type="radio" name="scheduleType" checked={form.scheduleType === st}
                    onChange={() => set("scheduleType", st)} className="accent-primary" />
                  {st === "manual" ? t("tasks.scheduleManual") :
                   st === "one-shot" ? t("tasks.scheduleOneShot") :
                   t("tasks.scheduleCron")}
                </label>
              ))}
            </div>
            {form.scheduleType === "one-shot" && (
              <div className="space-y-1">
                <Label className="text-xs">{t("tasks.runAtLabel")}</Label>
                <Input type="datetime-local" value={form.run_at?.replace("Z", "").slice(0, 16) ?? ""}
                  onChange={(e) => { const v = e.target.value; set("run_at", v ? new Date(v).toISOString() : ""); }} />
              </div>
            )}
            {form.scheduleType === "cron" && (
              <div className="space-y-1">
                <Label className="text-xs">{t("tasks.cronExpression")}</Label>
                <Input value={form.schedule} onChange={(e) => set("schedule", e.target.value)}
                  placeholder={t("tasks.cronPlaceholder")} className="font-mono" />
              </div>
            )}
          </div>

          <div className="space-y-1">
            <Label>{t("tasks.taskModel")}</Label>
            <select className={selectClass} value={form.model} onChange={(e) => set("model", e.target.value)}>
              <option value="">{t("tasks.taskModelDefault")}</option>
              {modelList.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <Label>{t("tasks.taskSkill")}</Label>
              <select className={selectClass} value={form.skill}
                disabled={!!form.workflow}
                onChange={(e) => set("skill", e.target.value)}>
                <option value="">{t("tasks.noneSelected")}</option>
                {skillList.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
              {form.workflow && (
                <p className="text-xs text-muted-foreground">{t("tasks.skillWorkflowExclusive")}</p>
              )}
            </div>
            <div className="space-y-1">
              <Label>{t("tasks.taskWorkflow")}</Label>
              <select className={selectClass} value={form.workflow}
                disabled={!!form.skill}
                onChange={(e) => set("workflow", e.target.value)}>
                <option value="">{t("tasks.noneSelected")}</option>
                {workflowList.map((w) => <option key={w} value={w}>{w}</option>)}
              </select>
              {form.skill && (
                <p className="text-xs text-muted-foreground">{t("tasks.skillWorkflowExclusive")}</p>
              )}
            </div>
          </div>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex justify-end">
            <Button type="submit" disabled={isPending || !form.name.trim() || !form.prompt.trim()}>
              {isPending ? t("tasks.saving") : mode === "create" ? t("tasks.createTask") : t("tasks.saveTask")}
            </Button>
          </div>
        </CardContent>
      </Card>
    </form>
  );
}
