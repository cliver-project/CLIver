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
import { RefInsertDropdown } from "@/components/lab/RefInsertDropdown";
import { useWebSocket } from "@/hooks/use-websocket";
import { useAgents } from "@/hooks/use-api";
import type { Cell } from "@/hooks/use-lab";
import { cn } from "@/lib/utils";
import { useTranslation } from "@/i18n";

interface LlmCellProps {
  cell: Cell;
  labId: string;
  runTrigger: number;
  onInputsChange: (inputs: Record<string, unknown>) => void;
  onExecutionComplete: (outputs: Record<string, unknown>, status: string, error?: string) => void;
}

interface TerminalLine {
  type: "text" | "thinking" | "tool" | "status" | "error";
  content: string;
  ts: number;
}

export function LlmCell({ cell, labId, runTrigger, onInputsChange, onExecutionComplete }: LlmCellProps) {
  const { t } = useTranslation();
  const { data: agents } = useAgents();
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);
  const [showVerification, setShowVerification] = useState(false);
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const onCompleteRef = useRef(onExecutionComplete);
  onCompleteRef.current = onExecutionComplete;

  const prompt = (cell.inputs.prompt as string) || "";
  const agent = (cell.inputs.agent as string) || "";
  const systemPrompt = (cell.inputs.system_prompt as string) || "";
  const outputFormat = (cell.inputs.output_format as string) || "text";

  // Verification config
  const verification = (cell.inputs.verification as Record<string, unknown>) || null;
  const maxRetries = verification ? Number(verification.max_retries || 3) : 3;
  const timeoutS = verification ? Number(verification.timeout_s || 300) : 300;
  const verifierAgent = verification ? String(verification.verifier_agent || "") : "";

  // WebSocket for streaming
  const wsUrl = isExecuting
    ? `/admin/ws/labs/${encodeURIComponent(labId)}/cells/${encodeURIComponent(cell.id)}`
    : null;
  const ws = useWebSocket(wsUrl);

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  // Handle WebSocket messages
  useEffect(() => {
    if (!ws.lastMessage) return;
    const msg = ws.lastMessage;

    if (msg.type === "text" && msg.content) {
      setLines((prev) => [...prev, { type: "text", content: msg.content as string, ts: Date.now() }]);
    } else if (msg.type === "thinking" && msg.content) {
      setLines((prev) => [...prev, { type: "thinking", content: msg.content as string, ts: Date.now() }]);
    } else if (msg.type === "tool" && msg.content) {
      setLines((prev) => [...prev, { type: "tool", content: msg.content as string, ts: Date.now() }]);
    } else if (msg.type === "status") {
      setLines((prev) => [...prev, { type: "status", content: `▶ ${msg.status || "running"}`, ts: Date.now() }]);
    } else if (msg.type === "done") {
      setIsExecuting(false);
      if (msg.outputs) {
        onCompleteRef.current?.(msg.outputs as Record<string, unknown>, "completed");
      }
      ws.close();
    } else if (msg.type === "error") {
      setIsExecuting(false);
      setLines((prev) => [...prev, { type: "error", content: msg.message || "Unknown error", ts: Date.now() }]);
      onCompleteRef.current?.({}, "error", msg.message || "Unknown error");
      ws.close();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws.lastMessage]);

  // Once connected, send execute command
  useEffect(() => {
    if (isExecuting && ws.isConnected) {
      ws.send({ action: "execute" });
    }
  }, [isExecuting, ws.isConnected, ws]);

  // ── Trigger execution from parent (CellSlide header Run button) ──

  useEffect(() => {
    if (runTrigger < 0) {
      // Stop signal
      setIsExecuting(false);
      ws.close();
      return;
    }
    if (runTrigger > 0) {
      setLines([]);
      setIsExecuting(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runTrigger]);

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
      }
    },
    [cell.inputs, onInputsChange, prompt],
  );

  const updateInput = useCallback(
    (key: string, value: unknown) => {
      onInputsChange({ ...cell.inputs, [key]: value });
    },
    [cell.inputs, onInputsChange],
  );

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
          <Label className="text-xs font-medium">{t("lab.agent")}</Label>
          <Select value={agent} onValueChange={(v) => updateInput("agent", v)} disabled={isExecuting}>
            <SelectTrigger className="mt-1 h-8 text-sm">
              <SelectValue placeholder={t("lab.selectAgent")} />
            </SelectTrigger>
            <SelectContent>
              {agentList.map((a) => (
                <SelectItem key={a} value={a}>{a}</SelectItem>
              ))}
              {agentList.length === 0 && (
                <SelectItem value="cliver" disabled>No agents configured</SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
        <div className="w-28">
          <Label className="text-xs font-medium">{t("lab.outputFormat")}</Label>
          <Select value={outputFormat} onValueChange={(v) => updateInput("output_format", v)} disabled={isExecuting}>
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
          <Label className="text-xs font-medium">{t("lab.prompt")}</Label>
          <RefInsertDropdown labId={labId} cellId={cell.id} onInsert={handleRefInsert} />
        </div>
        <textarea
          ref={textareaRef}
          value={prompt}
          onChange={(e) => updateInput("prompt", e.target.value)}
          disabled={isExecuting}
          placeholder={t("lab.promptPlaceholder")}
          className="w-full min-h-[120px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-1 disabled:opacity-60"
        />
      </div>

      {/* System prompt */}
      <div>
        <button
          type="button"
          onClick={() => setShowSystemPrompt(!showSystemPrompt)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {showSystemPrompt ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          {t("lab.systemPrompt")}
        </button>
        {showSystemPrompt && (
          <textarea
            value={systemPrompt}
            onChange={(e) => updateInput("system_prompt", e.target.value)}
            disabled={isExecuting}
            placeholder={t("lab.systemPromptPlaceholder")}
            className="w-full mt-1 min-h-[80px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-1 disabled:opacity-60"
          />
        )}
      </div>

      {/* Verification config */}
      <div>
        <button
          type="button"
          onClick={() => setShowVerification(!showVerification)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {showVerification ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          {t("lab.verification")}
        </button>
        {showVerification && (
          <div className="mt-2 space-y-2 pl-2 border-l-2 border-muted">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label className="text-[11px] text-muted-foreground">{t("lab.maxRetries")}</Label>
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={maxRetries}
                  onChange={(e) =>
                    updateInput("verification", {
                      ...verification,
                      max_retries: parseInt(e.target.value) || 0,
                    })
                  }
                  disabled={isExecuting}
                  className="w-full mt-0.5 rounded-md border border-input bg-background px-2 py-1 text-sm"
                />
              </div>
              <div>
                <Label className="text-[11px] text-muted-foreground">{t("lab.timeoutSeconds")}</Label>
                <input
                  type="number"
                  min={10}
                  max={3600}
                  value={timeoutS}
                  onChange={(e) =>
                    updateInput("verification", {
                      ...verification,
                      timeout_s: parseInt(e.target.value) || 300,
                    })
                  }
                  disabled={isExecuting}
                  className="w-full mt-0.5 rounded-md border border-input bg-background px-2 py-1 text-sm"
                />
              </div>
            </div>
            {verifierAgent !== undefined && (
              <div>
                <label className="text-[11px] text-muted-foreground">{t("lab.verifierAgent")}</label>
                <Select
                  value={verifierAgent}
                  onValueChange={(v) =>
                    updateInput("verification", { ...verification, verifier_agent: v === "__default__" ? "" : v })
                  }
                  disabled={isExecuting}
                >
                  <SelectTrigger className="mt-1 h-8 text-sm">
                    <SelectValue placeholder="(default)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__default__">(use default agent)</SelectItem>
                    {agentList.map((a) => (
                      <SelectItem key={a} value={a}>{a}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Terminal output */}
      {(lines.length > 0 || isExecuting) && (
        <div
          ref={terminalRef}
          className="rounded-lg bg-[#0d1117] border border-[#30363d] overflow-y-auto max-h-[400px] min-h-[120px]"
        >
          <div className="p-3 font-mono text-[13px] leading-relaxed space-y-0.5">
            {lines.map((line, i) => (
              <div key={i} className={cn("whitespace-pre-wrap break-words", lineColor(line.type))}>
                {line.content}
              </div>
            ))}
            {isExecuting && (
              <div className="flex items-center gap-1.5 text-[#8b949e]">
                <span className="inline-block w-2 h-4 bg-[#58a6ff] animate-pulse" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function lineColor(type: TerminalLine["type"]): string {
  switch (type) {
    case "thinking":
      return "text-[#8b949e] italic";
    case "tool":
      return "text-[#d2a8ff]";
    case "status":
      return "text-[#58a6ff] text-xs";
    case "error":
      return "text-[#f85149]";
    default:
      return "text-[#c9d1d9]";
  }
}
