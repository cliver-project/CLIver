import { useState, useCallback } from "react";
import { useParams } from "react-router";
import { useLab, useUpdateLab, useExecuteCell, useRunAll } from "@/hooks/use-lab";
import type { Cell } from "@/hooks/use-lab";
import { LabToolbar } from "@/components/lab/LabToolbar";
import { CellCard } from "@/components/lab/CellCard";
import { ConfigCell } from "@/components/lab/ConfigCell";
import { LlmCell } from "@/components/lab/LlmCell";
import { CodeCell } from "@/components/lab/CodeCell";
import { DisplayCell } from "@/components/lab/DisplayCell";
import { AddCellButton } from "@/components/lab/AddCellButton";
import { useTranslation } from "@/i18n";

export default function LabEditor() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const { data: lab, isLoading, error } = useLab(id || "");
  const updateLab = useUpdateLab(id || "");
  const executeCell = useExecuteCell(id || "");
  const runAll = useRunAll(id || "");
  const [expandedCells, setExpandedCells] = useState<Set<string>>(new Set());

  const toggleCell = useCallback((cellId: string) => {
    setExpandedCells((prev) => {
      const next = new Set(prev);
      if (next.has(cellId)) {
        next.delete(cellId);
      } else {
        next.add(cellId);
      }
      return next;
    });
  }, []);

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
        code: { source: 'def run(ctx):\n    # Access previous cell outputs: ctx.refs("cell_id.outputs.field")\n    \n    return {"result": "hello"}' },
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
      setExpandedCells((prev) => new Set(prev).add(cellId));
    },
    [lab, updateLab],
  );

  const handleSave = useCallback(() => {
    if (lab) {
      updateLab.mutate(lab);
    }
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
      const cells = lab.cells.map((c) =>
        c.id === cellId ? { ...c, inputs } : c,
      );
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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-sm text-muted-foreground">{t("labs.loading")}</div>
      </div>
    );
  }

  if (error || !lab) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-sm text-red-600">
          {error ? t("labs.loadError", { error: error.message }) : t("labs.notFound")}
        </div>
      </div>
    );
  }

  const overallStatus = lab.cells.some((c) => c.status === "error")
    ? "error"
    : lab.cells.some((c) => c.status === "running")
      ? "running"
      : lab.cells.every((c) => c.status === "completed")
        ? "completed"
        : "idle";

  return (
    <div className="max-w-4xl mx-auto">
      <LabToolbar
        title={lab.title}
        scenarioId={lab.scenario_id}
        cellCount={lab.cells.length}
        overallStatus={overallStatus}
        onRunAll={handleRunAll}
        onSave={handleSave}
        isRunning={runAll.isPending}
        isSaving={updateLab.isPending}
      />

      <div className="space-y-3">
        {lab.cells.map((cell, idx) => (
          <CellCard
            key={cell.id}
            cell={cell}
            isExpanded={expandedCells.has(cell.id)}
            onToggle={() => toggleCell(cell.id)}
            onExecute={() => handleExecuteCell(cell.id)}
            onDelete={() => handleDeleteCell(cell.id)}
            onMoveUp={() => handleMoveCell(cell.id, "up")}
            onMoveDown={() => handleMoveCell(cell.id, "down")}
            isFirst={idx === 0}
            isLast={idx === lab.cells.length - 1}
          >
            {cell.type === "config" && (
              <ConfigCell cell={cell} onSave={(outputs) => handleConfigSave(cell.id, outputs)} />
            )}
            {cell.type === "llm" && (
              <LlmCell
                cell={cell}
                labId={lab.id}
                onInputsChange={(inputs) => handleInputsChange(cell.id, inputs)}
                onExecutionComplete={(outputs, status, err) =>
                  handleExecutionComplete(cell.id, outputs, status, err)
                }
              />
            )}
            {cell.type === "code" && (
              <CodeCell
                cell={cell}
                onSourceChange={(source) => handleSourceChange(cell.id, source)}
              />
            )}
            {cell.type === "display" && <DisplayCell cell={cell} />}
          </CellCard>
        ))}

        <AddCellButton onAdd={handleAddCell} />
      </div>
    </div>
  );
}
