import { ArrowLeft, Save, Play, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useTranslation } from "@/i18n";

interface LabHeaderProps {
  title: string;
  scenarioId?: string | null;
  onBack: () => void;
  onRunAll: () => void;
  onSave: () => void;
  isRunning: boolean;
  isSaving: boolean;
}

export function LabHeader({
  title,
  scenarioId,
  onBack,
  onRunAll,
  onSave,
  isRunning,
  isSaving,
}: LabHeaderProps) {
  const { t } = useTranslation();
  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b bg-card shrink-0">
      <Button variant="ghost" size="sm" onClick={onBack} className="h-8 w-8 p-0">
        <ArrowLeft className="w-4 h-4" />
      </Button>

      <h1 className="text-base font-semibold truncate flex-1">{title}</h1>

      {scenarioId && (
        <Badge variant="secondary" className="text-xs">
          {scenarioId}
        </Badge>
      )}

      <Button variant="outline" size="sm" onClick={onRunAll} disabled={isRunning} className="gap-1.5 h-8">
        {isRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
        {t("lab.runAll")}
      </Button>

      <Button variant="default" size="sm" onClick={onSave} disabled={isSaving} className="gap-1.5 h-8">
        {isSaving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
        {t("lab.save")}
      </Button>
    </div>
  );
}
