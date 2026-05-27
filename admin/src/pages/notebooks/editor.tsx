import { useState, useCallback } from "react";
import { useParams } from "react-router";
import { useNotebook, useUpdateNotebook, useExecuteCell, useRunAll } from "@/hooks/use-notebook";
import type { Cell } from "@/hooks/use-notebook";
import { NotebookToolbar } from "@/components/notebook/NotebookToolbar";
import { CellCard } from "@/components/notebook/CellCard";
import { ConfigCell } from "@/components/notebook/ConfigCell";
import { LlmCell } from "@/components/notebook/LlmCell";
import { CodeCell } from "@/components/notebook/CodeCell";
import { DisplayCell } from "@/components/notebook/DisplayCell";
import { AddCellButton } from "@/components/notebook/AddCellButton";

export default function NotebookEditor() {
  const { id } = useParams<{ id: string }>();
  const { data: notebook, isLoading, error } = useNotebook(id || "");
  const updateNotebook = useUpdateNotebook(id || "");
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
      if (!notebook) return;
      const updatedCells = notebook.cells.filter((c) => c.id !== cellId);
      updateNotebook.mutate({ ...notebook, cells: updatedCells });
    },
    [notebook, updateNotebook],
  );

  const handleMoveCell = useCallback(
    (cellId: string, direction: "up" | "down") => {
      if (!notebook) return;
      const cells = [...notebook.cells];
      const idx = cells.findIndex((c) => c.id === cellId);
      if (idx < 0) return;
      const targetIdx = direction === "up" ? idx - 1 : idx + 1;
      if (targetIdx < 0 || targetIdx >= cells.length) return;
      const temp = cells[idx]!;
      cells[idx] = cells[targetIdx]!;
      cells[targetIdx] = temp;
      updateNotebook.mutate({ ...notebook, cells });
    },
    [notebook, updateNotebook],
  );

  const handleAddCell = useCallback(
    (type: string) => {
      if (!notebook) return;
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
      updateNotebook.mutate({ ...notebook, cells: [...notebook.cells, newCell] });
      setExpandedCells((prev) => new Set(prev).add(cellId));
    },
    [notebook, updateNotebook],
  );

  const handleSave = useCallback(() => {
    if (notebook) {
      updateNotebook.mutate(notebook);
    }
  }, [notebook, updateNotebook]);

  const handleRunAll = useCallback(() => {
    runAll.mutate();
  }, [runAll]);

  const handleConfigSave = useCallback(
    (cellId: string, outputs: Record<string, unknown>) => {
      if (!notebook) return;
      const cells = notebook.cells.map((c) =>
        c.id === cellId ? { ...c, outputs, status: "completed" as const } : c,
      );
      updateNotebook.mutate({ ...notebook, cells });
    },
    [notebook, updateNotebook],
  );

  const handleInputsChange = useCallback(
    (cellId: string, inputs: Record<string, unknown>) => {
      if (!notebook) return;
      const cells = notebook.cells.map((c) =>
        c.id === cellId ? { ...c, inputs } : c,
      );
      updateNotebook.mutate({ ...notebook, cells });
    },
    [notebook, updateNotebook],
  );

  const handleSourceChange = useCallback(
    (cellId: string, source: string) => {
      if (!notebook) return;
      const cells = notebook.cells.map((c) =>
        c.id === cellId ? { ...c, inputs: { ...c.inputs, source } } : c,
      );
      updateNotebook.mutate({ ...notebook, cells });
    },
    [notebook, updateNotebook],
  );

  const handleExecutionComplete = useCallback(
    (cellId: string, outputs: Record<string, unknown>, status: string, cellError?: string) => {
      if (!notebook) return;
      const validStatus = status as Cell["status"];
      const cells = notebook.cells.map((c) =>
        c.id === cellId ? { ...c, outputs, status: validStatus, error: cellError || null } : c,
      );
      updateNotebook.mutate({ ...notebook, cells });
    },
    [notebook, updateNotebook],
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-sm text-muted-foreground">Loading notebook...</div>
      </div>
    );
  }

  if (error || !notebook) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-sm text-red-600">
          {error ? `Failed to load notebook: ${error.message}` : "Notebook not found"}
        </div>
      </div>
    );
  }

  const overallStatus = notebook.cells.some((c) => c.status === "error")
    ? "error"
    : notebook.cells.some((c) => c.status === "running")
      ? "running"
      : notebook.cells.every((c) => c.status === "completed")
        ? "completed"
        : "idle";

  return (
    <div className="max-w-4xl mx-auto">
      <NotebookToolbar
        title={notebook.title}
        scenarioId={notebook.scenario_id}
        cellCount={notebook.cells.length}
        overallStatus={overallStatus}
        onRunAll={handleRunAll}
        onSave={handleSave}
        isRunning={runAll.isPending}
        isSaving={updateNotebook.isPending}
      />

      <div className="space-y-3">
        {notebook.cells.map((cell, idx) => (
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
            isLast={idx === notebook.cells.length - 1}
          >
            {cell.type === "config" && (
              <ConfigCell cell={cell} onSave={(outputs) => handleConfigSave(cell.id, outputs)} />
            )}
            {cell.type === "llm" && (
              <LlmCell
                cell={cell}
                notebookId={notebook.id}
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
