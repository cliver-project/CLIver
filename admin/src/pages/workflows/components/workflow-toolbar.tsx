import { ArrowLeft, Plus, Play, Save } from "lucide-react";
import { Link } from "react-router";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTranslation } from "@/i18n";

interface WorkflowToolbarProps {
  name: string;
  stepCount: number;
  onAddNode: (type: "llm" | "python") => void;
  onRun: () => void;
  onSave: () => void;
  saving?: boolean;
}

export function WorkflowToolbar({
  name,
  stepCount,
  onAddNode,
  onRun,
  onSave,
  saving,
}: WorkflowToolbarProps) {
  const { t } = useTranslation();

  return (
    <div className="h-10 bg-background/90 backdrop-blur border-b border-border flex items-center px-3 gap-2">
      <Link to="/admin/workflows">
        <Button variant="ghost" size="icon" className="w-7 h-7">
          <ArrowLeft className="w-4 h-4" />
        </Button>
      </Link>
      <span className="font-semibold text-sm">{name}</span>
      <span className="text-xs text-muted-foreground">{t("common.steps", { count: stepCount })}</span>

      <div className="flex-1" />

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm" className="h-7 text-xs">
            <Plus className="w-3 h-3 mr-1" /> {t("workflows.addNode")}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem onClick={() => onAddNode("llm")}>{t("workflows.llmStep")}</DropdownMenuItem>
          <DropdownMenuItem onClick={() => onAddNode("python")}>{t("workflows.pythonStep")}</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Button size="sm" className="h-7 text-xs bg-emerald-600 hover:bg-emerald-700" onClick={onRun}>
        <Play className="w-3 h-3 mr-1" /> {t("common.run")}
      </Button>

      <Button size="sm" className="h-7 text-xs" onClick={onSave} disabled={saving}>
        <Save className="w-3 h-3 mr-1" /> {saving ? t("workflows.saving") : t("common.save")}
      </Button>
    </div>
  );
}
