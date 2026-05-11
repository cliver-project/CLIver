import { ArrowLeft, Plus, Play, Save, Trash2, Check, Loader2, History } from "lucide-react";
import { Link } from "react-router";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { StatusPill } from "@/components/status-pill";
import { useTranslation } from "@/i18n";

interface WorkflowToolbarProps {
  name: string;
  stepCount: number;
  outputsDir?: string;
  onAddNode: (type: "llm" | "python") => void;
  onRun: () => void;
  onSave: () => void;
  onDelete?: () => void;
  saving?: boolean;
  saved?: boolean;
  running?: boolean;
  executions?: Array<Record<string, unknown>>;
  selectedExecutionId?: string | null;
  onSelectExecution?: (id: string) => void;
}

export function WorkflowToolbar({
  name,
  stepCount,
  onAddNode,
  onRun,
  onSave,
  outputsDir,
  onDelete,
  saving,
  saved,
  running,
  executions,
  selectedExecutionId,
  onSelectExecution,
}: WorkflowToolbarProps) {
  const { t } = useTranslation();
  const selectedExec = executions?.find((e) => String(e.thread_id) === selectedExecutionId);

  return (
    <div className="h-10 bg-background/90 backdrop-blur border-b border-border flex items-center px-3 gap-2">
      <Link to="/admin/workflows">
        <Button variant="ghost" size="icon" className="w-7 h-7">
          <ArrowLeft className="w-4 h-4" />
        </Button>
      </Link>
      <span className="font-semibold text-sm">{name}</span>
      <span className="text-xs text-muted-foreground">{t("common.steps", { count: stepCount })}</span>
      {outputsDir && (
        <span className="text-[10px] text-muted-foreground font-mono truncate max-w-[300px]" title={outputsDir}>
          {outputsDir}
        </span>
      )}

      {executions && executions.length > 0 && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm" className="h-7 text-xs ml-2">
              <History className="w-3 h-3 mr-1" />
              {selectedExec ? (
                <>
                  <StatusPill status={String(selectedExec.status ?? "unknown")} />
                  <span className="font-mono ml-1">{String(selectedExec.thread_id ?? "").slice(0, 8)}</span>
                </>
              ) : (
                t("workflows.executions")
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="z-[9999] max-h-64 overflow-y-auto bg-popover shadow-lg border">
            {executions.map((e) => {
              const tid = String(e.thread_id ?? "");
              const isSelected = tid === selectedExecutionId;
              return (
                <DropdownMenuItem
                  key={tid}
                  onClick={() => onSelectExecution?.(tid)}
                  className={isSelected ? "bg-accent" : ""}
                >
                  <div className="flex items-center gap-2">
                    <StatusPill status={String(e.status ?? "unknown")} />
                    <span className="font-mono text-xs">{tid.slice(0, 8)}</span>
                    <span className="text-[10px] text-muted-foreground">{String(e.started_at ?? "").slice(0, 19)}</span>
                  </div>
                </DropdownMenuItem>
              );
            })}
          </DropdownMenuContent>
        </DropdownMenu>
      )}

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

      <Button
        size="sm"
        className={running ? "h-7 text-xs bg-zinc-600 hover:bg-zinc-600 cursor-not-allowed" : "h-7 text-xs bg-emerald-600 hover:bg-emerald-700"}
        onClick={onRun}
        disabled={running}
      >
        {running ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Play className="w-3 h-3 mr-1" />}
        {running ? t("common.running") : t("common.run")}
      </Button>

      <Button
        size="sm"
        className={saved ? "h-7 text-xs bg-emerald-600 hover:bg-emerald-700" : "h-7 text-xs"}
        onClick={onSave}
        disabled={saving}
      >
        {saved ? <Check className="w-3 h-3 mr-1" /> : <Save className="w-3 h-3 mr-1" />}
        {saving ? t("workflows.saving") : saved ? t("workflows.saved") : t("common.save")}
      </Button>

      {onDelete && (
        <Button size="sm" variant="destructive" className="h-7 text-xs" onClick={onDelete}>
          <Trash2 className="w-3 h-3 mr-1" /> {t("common.delete")}
        </Button>
      )}
    </div>
  );
}
