import { useCallback, useEffect, useRef, useState } from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { RefInsertDropdown } from "@/components/notebook/RefInsertDropdown";
import { CellOutput } from "@/components/notebook/CellOutput";
import { useWebSocket } from "@/hooks/use-websocket";
import { useAgents } from "@/hooks/use-api";
import type { Cell } from "@/hooks/use-notebook";
import { cn } from "@/lib/utils";

interface LlmCellProps {
  cell: Cell;
  notebookId: string;
  onInputsChange: (inputs: Record<string, unknown>) => void;
  onExecutionComplete: (outputs: Record<string, unknown>, status: string, error?: string) => void;
}

export function LlmCell({ cell, notebookId, onInputsChange, onExecutionComplete }: LlmCellProps) {
  const { data: agents } = useAgents();
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [isExecuting, setIsExecuting] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const prompt = (cell.inputs.prompt as string) || "";
  const agent = (cell.inputs.agent as string) || "";
  const systemPrompt = (cell.inputs.system_prompt as string) || "";
  const outputFormat = (cell.inputs.output_format as string) || "text";

  // WebSocket for streaming
  const wsUrl = isExecuting
    ? `/admin/ws/notebooks/${encodeURIComponent(notebookId)}/cells/${encodeURIComponent(cell.id)}`
    : null;
  const ws = useWebSocket(wsUrl);

  // Handle WebSocket messages
  useEffect(() => {
    if (!ws.lastMessage) return;
    const msg = ws.lastMessage;

    if (msg.type === "chunk" && msg.text) {
      setStreamingText((prev) => prev + msg.text);
    } else if (msg.type === "done") {
      setIsExecuting(false);
      setStreamingText("");
      if (msg.outputs) {
        onExecutionComplete(msg.outputs as Record<string, unknown>, "completed");
      }
      ws.close();
    } else if (msg.type === "error") {
      setIsExecuting(false);
      setStreamingText("");
      onExecutionComplete({}, "error", msg.message || "Unknown error");
      ws.close();
    }
  }, [ws.lastMessage, ws, onExecutionComplete]);

  // Once connected, send execute command
  useEffect(() => {
    if (isExecuting && ws.isConnected) {
      ws.send({ action: "execute" });
    }
  }, [isExecuting, ws.isConnected, ws]);

  // Insert reference at cursor position in prompt textarea
  const handleRefInsert = useCallback(
    (ref: string) => {
      const ta = textareaRef.current;
      if (ta) {
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        const newPrompt = prompt.substring(0, start) + ref + prompt.substring(end);
        onInputsChange({ ...cell.inputs, prompt: newPrompt });
        requestAnimationFrame(() => {
          ta.focus();
          ta.setSelectionRange(start + ref.length, start + ref.length);
        });
      } else {
        onInputsChange({ ...cell.inputs, prompt: prompt + ref });
      }
    },
    [prompt, cell.inputs, onInputsChange],
  );

  const updateInput = (key: string, value: string) => {
    onInputsChange({ ...cell.inputs, [key]: value });
  };

  // Extract agent names from the array of agent objects
  const agentList: string[] = agents
    ? (agents as Array<Record<string, unknown>>)
        .map((a) => a.name as string)
        .filter((name): name is string => !!name)
    : [];

  return (
    <div className="space-y-3">
      {/* Agent + Output format row */}
      <div className="flex gap-3">
        <div className="flex-1">
          <Label className="text-xs font-medium">Agent</Label>
          <Select value={agent} onValueChange={(v) => updateInput("agent", v)}>
            <SelectTrigger className="mt-1 h-8 text-sm">
              <SelectValue placeholder="Select agent" />
            </SelectTrigger>
            <SelectContent>
              {agentList.map((a) => (
                <SelectItem key={a} value={a}>
                  {a}
                </SelectItem>
              ))}
              {agentList.length === 0 && (
                <SelectItem value="cliver" disabled>
                  No agents configured
                </SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
        <div className="w-28">
          <Label className="text-xs font-medium">Output</Label>
          <Select value={outputFormat} onValueChange={(v) => updateInput("output_format", v)}>
            <SelectTrigger className="mt-1 h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="text">Text</SelectItem>
              <SelectItem value="json">JSON</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Prompt */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <Label className="text-xs font-medium">Prompt</Label>
          <RefInsertDropdown
            notebookId={notebookId}
            cellId={cell.id}
            onInsert={handleRefInsert}
          />
        </div>
        <textarea
          ref={textareaRef}
          value={prompt}
          onChange={(e) => updateInput("prompt", e.target.value)}
          placeholder="Enter your prompt... Use 'Insert Ref' to reference previous cell outputs."
          rows={4}
          className={cn(
            "border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:ring-ring flex min-h-[60px] w-full rounded-md border px-3 py-2 text-base shadow-xs focus-visible:ring-1 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
            "text-sm font-mono resize-y"
          )}
          disabled={isExecuting}
        />
      </div>

      {/* System prompt (collapsible) */}
      <div>
        <button
          onClick={() => setShowSystemPrompt(!showSystemPrompt)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {showSystemPrompt ? (
            <ChevronDown className="w-3 h-3" />
          ) : (
            <ChevronRight className="w-3 h-3" />
          )}
          System Prompt (optional)
        </button>
        {showSystemPrompt && (
          <textarea
            value={systemPrompt}
            onChange={(e) => updateInput("system_prompt", e.target.value)}
            placeholder="Instructions for the AI agent..."
            rows={2}
            className={cn(
              "border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:ring-ring flex min-h-[60px] w-full rounded-md border px-3 py-2 text-base shadow-xs focus-visible:ring-1 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
              "mt-1 text-sm resize-y"
            )}
            disabled={isExecuting}
          />
        )}
      </div>

      {/* Streaming output */}
      {isExecuting && (
        <div className="rounded-lg bg-muted/50 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
            <Loader2 className="w-3 h-3 animate-spin" />
            Generating response...
          </div>
          {streamingText && (
            <div className="text-sm text-foreground whitespace-pre-wrap">
              {streamingText}
              <span className="inline-block w-1.5 h-4 bg-primary animate-pulse ml-0.5" />
            </div>
          )}
        </div>
      )}

      {/* Completed output */}
      {!isExecuting && (
        <CellOutput outputs={cell.outputs} error={cell.error} status={cell.status} />
      )}
    </div>
  );
}
