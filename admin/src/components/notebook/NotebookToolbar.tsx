import { useNavigate } from "react-router";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Play, Save, Loader2 } from "lucide-react";
import { CellStatusBadge } from "@/components/notebook/CellStatusBadge";

interface NotebookToolbarProps {
  title: string;
  scenarioId?: string | null;
  cellCount: number;
  overallStatus: string;
  onRunAll: () => void;
  onSave: () => void;
  isRunning: boolean;
  isSaving: boolean;
}

export function NotebookToolbar({
  title,
  scenarioId,
  cellCount,
  overallStatus,
  onRunAll,
  onSave,
  isRunning,
  isSaving,
}: NotebookToolbarProps) {
  const navigate = useNavigate();

  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate("/admin/notebooks")}
          className="h-8 w-8 p-0"
        >
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <div>
          <h1 className="text-lg font-semibold text-foreground">{title}</h1>
          <div className="flex items-center gap-2 mt-0.5">
            {scenarioId && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                {scenarioId}
              </span>
            )}
            <span className="text-xs text-muted-foreground">{cellCount} cells</span>
            <CellStatusBadge status={overallStatus} />
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="default"
          size="sm"
          onClick={onRunAll}
          disabled={isRunning || cellCount === 0}
          className="bg-emerald-600 hover:bg-emerald-700 text-white"
        >
          {isRunning ? (
            <>
              <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 mr-1.5" />
              Run All
            </>
          )}
        </Button>
        <Button variant="outline" size="sm" onClick={onSave} disabled={isSaving}>
          {isSaving ? (
            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
          ) : (
            <Save className="w-3.5 h-3.5 mr-1.5" />
          )}
          Save
        </Button>
      </div>
    </div>
  );
}
