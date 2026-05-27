import { useCallback, useState } from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChevronDown, ChevronRight, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatPanel } from "@/components/lab/ChatPanel";
import { useAgents } from "@/hooks/use-api";
import type { Cell } from "@/hooks/use-lab";
import { useTranslation } from "@/i18n";

interface LlmCellProps {
  cell: Cell;
  labId: string;
  onInputsChange: (inputs: Record<string, unknown>) => void;
  onSaveResult: (outputs: Record<string, unknown>, status: string) => void;
}

export function LlmCell({ cell, labId, onInputsChange, onSaveResult }: LlmCellProps) {
  const { t } = useTranslation();
  const { data: agents } = useAgents();
  const [runCount, setRunCount] = useState(0);

  const agent = (cell.inputs.agent as string) || "";
  const systemPrompt = (cell.inputs.system_prompt as string) || "";
  const outputFormat = (cell.inputs.output_format as string) || "text";
  const initialPrompt = (cell.inputs.prompt as string) || "";

  const agentList: string[] = agents
    ? (agents as Array<Record<string, unknown>>)
        .map((a) => a.name as string)
        .filter((name): name is string => !!name)
    : [];

  const updateInput = useCallback(
    (key: string, value: unknown) => {
      onInputsChange({ ...cell.inputs, [key]: value });
    },
    [cell.inputs, onInputsChange],
  );

  const handleSaveResult = useCallback(
    (text: string) => {
      onSaveResult({ text }, "completed");
    },
    [onSaveResult],
  );

  const handleRun = useCallback(() => {
    // Persist current config first, then trigger run
    onInputsChange({ ...cell.inputs });
    setRunCount((n) => n + 1);
  }, [cell.inputs, onInputsChange]);

  return (
    <div className="grid grid-cols-[280px_1fr] h-full gap-0">
      {/* Left Panel — Configuration */}
      <div className="border-r bg-card/50 flex flex-col">
        <div className="px-4 py-3 space-y-4 flex-1 overflow-y-auto">
          {/* Agent */}
          <div>
            <Label className="text-xs font-medium">{t("lab.agent")}</Label>
            <Select value={agent} onValueChange={(v) => updateInput("agent", v)}>
              <SelectTrigger className="mt-1 h-8 text-sm w-full">
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

          {/* Output Format */}
          <div>
            <Label className="text-xs font-medium">{t("lab.outputFormat")}</Label>
            <Select value={outputFormat} onValueChange={(v) => updateInput("output_format", v)}>
              <SelectTrigger className="mt-1 h-8 text-sm w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="text">Text</SelectItem>
                <SelectItem value="json">JSON</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* System Prompt — collapsible */}
          <SystemPromptSection
            value={systemPrompt}
            onChange={(v) => updateInput("system_prompt", v)}
            t={t}
          />
        </div>

        {/* Run button */}
        <div className="px-4 py-3 border-t">
          <Button
            variant="default"
            size="sm"
            className="w-full gap-1.5"
            onClick={handleRun}
            disabled={!initialPrompt}
          >
            <Play className="w-3.5 h-3.5" />
            <span className="text-xs">{t("lab.run")}</span>
          </Button>
        </div>
      </div>

      {/* Right Panel — Chat */}
      <div className="flex flex-col min-h-0">
        <ChatPanel
          labId={labId}
          cellId={cell.id}
          agent={agent}
          systemPrompt={systemPrompt}
          outputFormat={outputFormat}
          initialPrompt={initialPrompt}
          runTrigger={runCount}
          onSaveResult={handleSaveResult}
        />
      </div>
    </div>
  );
}

function SystemPromptSection({
  value,
  onChange,
  t,
}: {
  value: string;
  onChange: (v: string) => void;
  t: (key: string) => string;
}) {
  const [show, setShow] = useState(false);

  return (
    <div>
      <button
        type="button"
        onClick={() => setShow(!show)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {show ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {t("lab.systemPrompt")}
      </button>
      {show && (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={t("lab.systemPromptPlaceholder")}
          className="w-full mt-1 min-h-[80px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-1"
        />
      )}
    </div>
  );
}
