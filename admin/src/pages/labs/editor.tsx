import { useState, useCallback, useMemo } from "react";
import { useParams, useNavigate } from "react-router";
import { useLab, useUpdateLab, useExecuteCell, useRunAll } from "@/hooks/use-lab";
import type { Cell } from "@/hooks/use-lab";
import { LabHeader } from "@/components/lab/LabHeader";
import { LabProgress } from "@/components/lab/LabProgress";
import { CellSlide } from "@/components/lab/CellSlide";
import { ConfigCell } from "@/components/lab/ConfigCell";
import { LlmCell } from "@/components/lab/LlmCell";
import { CodeCell } from "@/components/lab/CodeCell";
import { DisplayCell } from "@/components/lab/DisplayCell";
import { AddCellButton } from "@/components/lab/AddCellButton";
import { CellErrorBoundary } from "@/components/lab/CellErrorBoundary";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, Plus } from "lucide-react";
import { useTranslation } from "@/i18n";

export default function LabEditor() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: lab, isLoading, error } = useLab(id || "");
  const updateLab = useUpdateLab(id || "");
  const executeCell = useExecuteCell(id || "");
  const runAll = useRunAll(id || "");
  const [currentIndex, setCurrentIndex] = useState(0);

  // Reset index when switching labs or when cells are added/removed
  const safeIndex = useMemo(() => {
    if (!lab) return 0;
    if (currentIndex >= lab.cells.length) return Math.max(0, lab.cells.length - 1);
    return currentIndex;
  }, [currentIndex, lab]);

  const handleExecuteCell = useCallback(
    (cellId: string) => {
      executeCell.mutate(cellId);
    },
    [executeCell],
  );

  const handleDeleteCell = useCallback(
    (cellId: string) => {
      if (!lab) return;
      const updatedCells = lab.cells.filter((c) => c.id !== cellId);
      updateLab.mutate({ ...lab, cells: updatedCells });
    },
    [lab, updateLab],
  );

  const handleMoveCell = useCallback(
    (cellId: string, direction: "up" | "down") => {
      if (!lab) return;
      const cells = [...lab.cells];
      const idx = cells.findIndex((c) => c.id === cellId);
      if (idx < 0) return;
      const targetIdx = direction === "up" ? idx - 1 : idx + 1;
      if (targetIdx < 0 || targetIdx >= cells.length) return;
      const temp = cells[idx]!;
      cells[idx] = cells[targetIdx]!;
      cells[targetIdx] = temp;
      updateLab.mutate({ ...lab, cells });
      setCurrentIndex(targetIdx);
    },
    [lab, updateLab],
  );

  const handleAddCell = useCallback(
    (type: string) => {
      if (!lab) return;
      const cellId = `cell_${Date.now().toString(36)}`;
      const defaults: Record<string, Record<string, unknown>> = {
        config: { schema: {} },
        llm: { prompt: "", agent: "", output_format: "text" },
        code: {
          source:
            'def run(ctx):\n    # Access previous cell outputs: ctx.refs("cell_id.outputs.field")\n    \n    return {"result": "hello"}',
        },
        display: { content: "", format: "markdown" },
      };
      const newCell: Cell = {
        id: cellId,
        type: type as Cell["type"],
        title: `New ${type.charAt(0).toUpperCase() + type.slice(1)} Cell`,
        inputs: defaults[type] || {},
        outputs: {},
        status: "idle",
        error: null,
        duration_ms: 0,
      };
      updateLab.mutate({ ...lab, cells: [...lab.cells, newCell] });
      setCurrentIndex(lab.cells.length);
    },
    [lab, updateLab],
  );

  const handleSave = useCallback(() => {
    if (lab) updateLab.mutate(lab);
  }, [lab, updateLab]);

  const handleRunAll = useCallback(() => {
    runAll.mutate();
  }, [runAll]);

  const handleConfigSave = useCallback(
    (cellId: string, outputs: Record<string, unknown>) => {
      if (!lab) return;
      const cells = lab.cells.map((c) =>
        c.id === cellId ? { ...c, outputs, status: "completed" as const } : c,
      );
      updateLab.mutate({ ...lab, cells });
    },
    [lab, updateLab],
  );

  const handleInputsChange = useCallback(
    (cellId: string, inputs: Record<string, unknown>) => {
      if (!lab) return;
      const cells = lab.cells.map((c) => (c.id === cellId ? { ...c, inputs } : c));
      updateLab.mutate({ ...lab, cells });
    },
    [lab, updateLab],
  );

  const handleSourceChange = useCallback(
    (cellId: string, source: string) => {
      if (!lab) return;
      const cells = lab.cells.map((c) =>
        c.id === cellId ? { ...c, inputs: { ...c.inputs, source } } : c,
      );
      updateLab.mutate({ ...lab, cells });
    },
    [lab, updateLab],
  );

  const handleExecutionComplete = useCallback(
    (cellId: string, outputs: Record<string, unknown>, status: string, cellError?: string) => {
      if (!lab) return;
      const validStatus = status as Cell["status"];
      const cells = lab.cells.map((c) =>
        c.id === cellId ? { ...c, outputs, status: validStatus, error: cellError || null } : c,
      );
      updateLab.mutate({ ...lab, cells });
    },
    [lab, updateLab],
  );

  const goNext = useCallback(() => {
    if (lab && safeIndex < lab.cells.length - 1) setCurrentIndex(safeIndex + 1);
  }, [lab, safeIndex]);

  const goPrev = useCallback(() => {
    if (safeIndex > 0) setCurrentIndex(safeIndex - 1);
  }, [safeIndex]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-sm text-muted-foreground">{t("labs.loading")}</div>
      </div>
    );
  }

  if (error || !lab) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-sm text-red-600">
          {error ? t("labs.loadError", { error: error.message }) : t("labs.notFound")}
        </div>
      </div>
    );
  }

  const cell = lab.cells[safeIndex];

  return (
    <div className="flex flex-col -m-6 h-[calc(100vh-3rem)]">
      {/* Top bar */}
      <LabHeader
        title={lab.title}
        scenarioId={lab.scenario_id}
        onBack={() => navigate("/admin/labs")}
        onRunAll={handleRunAll}
        onSave={handleSave}
        isRunning={runAll.isPending}
        isSaving={updateLab.isPending}
      />

      {/* Progress bar */}
      <LabProgress cells={lab.cells} currentIndex={safeIndex} onSelect={setCurrentIndex} />

      {/* Slide area — fills remaining height */}
      <div className="flex-1 flex flex-col min-h-0">
        {cell ? (
          <CellSlide
            key={cell.id}
            cell={cell}
            index={safeIndex}
            total={lab.cells.length}
            onExecute={
              cell.type === "llm"
                ? undefined
                : cell.type !== "display"
                  ? () => handleExecuteCell(cell.id)
                  : undefined
            }
            onSave={
              cell.type === "config" ? () => handleConfigSave(cell.id, cell.outputs)
              : cell.type === "llm" ? () => updateLab.mutate(lab)
              : undefined
            }
            isSaving={updateLab.isPending}
            isExecuting={executeCell.isPending || runAll.isPending}
            onStop={undefined}
            onDelete={() => handleDeleteCell(cell.id)}
            onMoveUp={() => handleMoveCell(cell.id, "up")}
            onMoveDown={() => handleMoveCell(cell.id, "down")}
            isFirst={safeIndex === 0}
            isLast={safeIndex === lab.cells.length - 1}
          >
            <CellErrorBoundary cellTitle={cell.title}>
              {cell.type === "config" && (
                <ConfigCell cell={cell} onSave={(outputs) => handleConfigSave(cell.id, outputs)} />
              )}
              {cell.type === "llm" && (
                <LlmCell
                  cell={cell}
                  labId={lab.id}
                  onInputsChange={(inputs) => handleInputsChange(cell.id, inputs)}
                  onSaveResult={(outputs, status) => handleExecutionComplete(cell.id, outputs, status)}
                />
              )}
              {cell.type === "code" && (
                <CodeCell cell={cell} onSourceChange={(source) => handleSourceChange(cell.id, source)} />
              )}
              {cell.type === "display" && <DisplayCell cell={cell} />}
            </CellErrorBoundary>
          </CellSlide>
        ) : (
          /* Empty state when no cells */
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-3">
              <p className="text-sm text-muted-foreground">{t("labs.noCells")}</p>
              <AddCellButton onAdd={handleAddCell} />
            </div>
          </div>
        )}
      </div>

      {/* Bottom nav: prev/next + add */}
      {lab.cells.length > 0 && (
        <div className="flex items-center justify-between px-4 py-2 border-t bg-card shrink-0">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={goPrev} disabled={safeIndex === 0} className="h-8 w-8 p-0">
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <span className="text-xs text-muted-foreground tabular-nums min-w-[48px] text-center">
              {safeIndex + 1} / {lab.cells.length}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={goNext}
              disabled={safeIndex >= lab.cells.length - 1}
              className="h-8 w-8 p-0"
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>

          <AddCellButton onAdd={handleAddCell} trigger={<AddCellTrigger />} />
        </div>
      )}
    </div>
  );
}

/** Simple plus-button trigger for the AddCellButton when in the bottom bar. */
function AddCellTrigger() {
  return (
    <Button variant="ghost" size="sm" className="h-8 gap-1.5">
      <Plus className="w-4 h-4" />
      <span className="text-xs">Add Cell</span>
    </Button>
  );
}
