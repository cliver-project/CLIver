import { useState, useEffect, useRef, useCallback } from "react";
import { useModels, useSkills } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

interface LabConfigPanelProps {
  selectedModel: string;
  systemPrompt: string;
  selectedSkills: string[];
  onModelChange: (v: string) => void;
  onSystemPromptChange: (v: string) => void;
  onSkillsChange: (v: string[]) => void;
  onSave: () => void;
  saving: boolean;
}

export function LabConfigPanel({
  selectedModel,
  systemPrompt,
  selectedSkills,
  onModelChange,
  onSystemPromptChange,
  onSkillsChange,
  onSave,
  saving,
}: LabConfigPanelProps) {
  const { t } = useTranslation();
  const { data: modelsData } = useModels();
  const { data: skills } = useSkills();

  const [localSysMsg, setLocalSysMsg] = useState(systemPrompt);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setLocalSysMsg(systemPrompt);
  }, [systemPrompt]);

  const handleSysMsgChange = useCallback((v: string) => {
    setLocalSysMsg(v);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => onSystemPromptChange(v), 400);
  }, [onSystemPromptChange]);

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  const modelList: string[] = modelsData?.models || [];
  const skillList: string[] = skills
    ? (skills as Array<Record<string, unknown>>).map((s) => s.name as string).filter(Boolean)
    : [];

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-y-auto space-y-4 p-3">
        {/* Model */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.model")}
          </label>
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            className="mt-1 w-full rounded-md border border-input bg-background px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="">{t("lab.modelPlaceholder")}</option>
            {modelList.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        {/* System Prompt */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.systemPrompt")}
          </label>
          <textarea
            value={localSysMsg}
            onChange={(e) => handleSysMsgChange(e.target.value)}
            rows={4}
            className="mt-1 w-full rounded-md border border-input bg-background px-2 py-1.5 text-xs resize-y focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder={t("lab.systemPromptPlaceholder")}
          />
        </div>

        {/* MCP Servers (placeholder) */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.mcpServers")}
          </label>
          <p className="text-[11px] text-muted-foreground mt-1">{t("lab.comingSoon")}</p>
        </div>

        {/* Knowledge Bases (placeholder) */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.knowledgeBases")}
          </label>
          <p className="text-[11px] text-muted-foreground mt-1">{t("lab.comingSoon")}</p>
        </div>

        {/* Skills */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.skills")}
          </label>
          {skillList.length === 0 ? (
            <p className="text-[11px] text-muted-foreground mt-1">{t("common.loading")}</p>
          ) : (
            <div className="mt-1 flex flex-wrap gap-1">
              {skillList.map((s) => {
                const active = selectedSkills.includes(s);
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() =>
                      onSkillsChange(
                        active ? selectedSkills.filter((x) => x !== s) : [...selectedSkills, s],
                      )
                    }
                    className={`rounded-md px-2 py-0.5 text-[11px] transition-colors ${
                      active
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-muted-foreground hover:bg-secondary"
                    }`}
                  >
                    {s}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Save button */}
      <div className="p-3 border-t shrink-0">
        <button
          type="button"
          onClick={onSave}
          disabled={saving}
          className="w-full rounded-md bg-primary text-primary-foreground px-3 py-1.5 text-xs font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {saving ? t("lab.configSaved") : t("lab.saveConfig")}
        </button>
      </div>
    </div>
  );
}
