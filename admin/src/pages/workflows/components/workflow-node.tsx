import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { cn } from "@/lib/utils";

export interface WorkflowNodeData {
  stepId: string;
  type: "llm" | "python";
  model?: string;
  role?: string;
  outputFormat?: string;
  prompt?: string;
  file?: string;
  status?: "pending" | "running" | "completed" | "failed";
  [key: string]: unknown;
}

export type WorkflowNodeType = Node<WorkflowNodeData, "workflowNode">;

function WorkflowNodeComponent({ data, selected }: NodeProps<WorkflowNodeType>) {
  const isLlm = data.type === "llm";
  const statusColors: Record<string, string> = {
    running: "border-amber-500 shadow-amber-500/20",
    completed: "border-emerald-500",
    failed: "border-red-500",
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
      <Handle type="target" position={Position.Top} className="!bg-primary !w-2 !h-2" />

      <div className="text-[10px] font-medium mb-0.5" style={{ color: isLlm ? "#818cf8" : "#34d399" }}>
        {isLlm ? "LLM" : "Python"}
      </div>
      <div className="font-semibold text-sm text-foreground">{data.stepId}</div>
      {isLlm && data.model && (
        <div className="text-[10px] text-muted-foreground mt-0.5">
          {data.model}{data.outputFormat ? ` · ${data.outputFormat}` : ""}
        </div>
      )}
      {!isLlm && data.file && (
        <div className="text-[10px] text-muted-foreground mt-0.5 font-mono">{data.file}</div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-primary !w-2 !h-2" />
    </div>
  );
}

export const WorkflowNode = memo(WorkflowNodeComponent);
