import { memo, useState, useRef, useEffect } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { cn } from "@/lib/utils";

export interface WorkflowNodeData {
  stepId: string;
  type: "llm" | "python";
  agent?: string;
  outputFormat?: string;
  prompt?: string;
  file?: string;
  code?: string;
  status?: "pending" | "running" | "completed" | "failed";
  onRename?: (oldId: string, newId: string) => void;
  [key: string]: unknown;
}

export type WorkflowNodeType = Node<WorkflowNodeData, "workflowNode">;

const handleClass = "!bg-primary !w-1.5 !h-1.5";

function WorkflowNodeComponent({ data, selected }: NodeProps<WorkflowNodeType>) {
  const isLlm = data.type === "llm";
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(data.stepId);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const commitRename = () => {
    setEditing(false);
    const trimmed = editValue.trim().replace(/[^a-z0-9_]/g, "_");
    if (trimmed && trimmed !== data.stepId && data.onRename) {
      data.onRename(data.stepId, trimmed);
    }
    setEditValue(trimmed || data.stepId);
  };

  const statusColors: Record<string, string> = {
    running: "border-amber-500 shadow-amber-500/20",
    completed: "border-emerald-500",
    failed: "border-red-500",
    suspended: "border-yellow-500",
    pending: "border-border",
  };

  return (
    <div
      className={cn(
        "px-4 py-2.5 rounded-lg bg-card border-2 min-w-[140px] transition-all",
        selected ? "border-primary shadow-lg shadow-primary/10" : statusColors[data.status ?? "pending"],
        data.status === "running" && "animate-pulse",
      )}
    >
      <Handle type="target" position={Position.Top} id="top-target" className={handleClass} />
      <Handle type="source" position={Position.Top} id="top-source" className={handleClass} />
      <Handle type="target" position={Position.Left} id="left-target" className={handleClass} />
      <Handle type="source" position={Position.Left} id="left-source" className={handleClass} />

      <div className="text-[10px] font-medium mb-0.5" style={{ color: isLlm ? "#818cf8" : "#34d399" }}>
        {isLlm ? "LLM" : "Python"}
      </div>

      {editing ? (
        <input
          ref={inputRef}
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={commitRename}
          onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") { setEditing(false); setEditValue(data.stepId); } }}
          className="font-semibold text-sm bg-transparent border-b border-primary outline-none w-full text-foreground"
        />
      ) : (
        <div
          className="font-semibold text-sm text-foreground cursor-text"
          onDoubleClick={() => { setEditValue(data.stepId); setEditing(true); }}
        >
          {data.stepId}
        </div>
      )}

      {isLlm && (data.agent || data.outputFormat) && (
        <div className="text-[10px] text-muted-foreground mt-0.5">
          {data.agent ?? "default"}{data.outputFormat ? ` · ${data.outputFormat}` : ""}
        </div>
      )}
      {!isLlm && data.file && (
        <div className="text-[10px] text-muted-foreground mt-0.5 font-mono">{data.file}</div>
      )}

      <Handle type="target" position={Position.Bottom} id="bottom-target" className={handleClass} />
      <Handle type="source" position={Position.Bottom} id="bottom-source" className={handleClass} />
      <Handle type="target" position={Position.Right} id="right-target" className={handleClass} />
      <Handle type="source" position={Position.Right} id="right-source" className={handleClass} />
    </div>
  );
}

export const WorkflowNode = memo(WorkflowNodeComponent);
