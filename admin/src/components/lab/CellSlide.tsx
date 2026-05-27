import { useRef, useEffect, useState } from "react";
import { Settings, Bot, Code2, FileText, Play, Save, Loader2, Square, Trash2, ArrowUp, ArrowDown, MoreHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { CellStatusBadge } from "@/components/lab/CellStatusBadge";
import { CellOutput } from "@/components/lab/CellOutput";
import { cn } from "@/lib/utils";
import type { Cell } from "@/hooks/use-lab";
import { useTranslation } from "@/i18n";

const TYPE_CONFIG: Record<string, { icon: React.ComponentType<{ className?: string }>; color: string; bgColor: string; labelKey: string }> = {
  config: { icon: Settings, color: "text-indigo-600", bgColor: "bg-indigo-50 dark:bg-indigo-950", labelKey: "lab.config" },
  llm: { icon: Bot, color: "text-purple-600", bgColor: "bg-purple-50 dark:bg-purple-950", labelKey: "lab.llm" },
  code: { icon: Code2, color: "text-emerald-600", bgColor: "bg-emerald-50 dark:bg-emerald-950", labelKey: "lab.code" },
  display: { icon: FileText, color: "text-amber-600", bgColor: "bg-amber-50 dark:bg-amber-950", labelKey: "lab.display" },
};

interface TerminalLine {
  type: "text" | "thinking" | "tool" | "status" | "error";
  content: string;
}

interface CellSlideProps {
  cell: Cell;
  index: number;
  total: number;
  onExecute?: () => void;
  onSave?: () => void;
  onStop?: () => void;
  isSaving?: boolean;
  isExecuting?: boolean;
  terminalLines?: TerminalLine[];
  isStreaming?: boolean;
  onDelete: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  isFirst: boolean;
  isLast: boolean;
  children: React.ReactNode;
}

export function CellSlide({
  cell,
  index,
  total,
  onExecute,
  onSave,
  onStop,
  isSaving,
  isExecuting,
  terminalLines,
  isStreaming,
  onDelete,
  onMoveUp,
  onMoveDown,
  isFirst,
  isLast,
  children,
}: CellSlideProps) {
  const { t } = useTranslation();
  const cfg = TYPE_CONFIG[cell.type] || TYPE_CONFIG.display;
  const Icon = cfg.icon;
  const isRunning = cell.status === "running";
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [terminalLines]);

  return (
    <div className="flex flex-col h-full">
      {/* Cell header — unified action bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b shrink-0 bg-card/80">
        <span className="text-[11px] text-muted-foreground font-mono tabular-nums w-10 shrink-0">
          {index + 1}/{total}
        </span>

        <div className={cn("w-5 h-5 rounded flex items-center justify-center shrink-0", cfg.bgColor)}>
          <Icon className={cn("w-3 h-3", cfg.color)} />
        </div>

        <span className="text-sm font-medium truncate flex-1">
          {cell.title || `${t(cfg.labelKey)} Cell`}
        </span>

        <CellStatusBadge status={cell.status} />

        {/* Save button — config and LLM cells */}
        {(cell.type === "config" || cell.type === "llm") && onSave && (
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 gap-1 transition-all"
            onClick={onSave}
            disabled={isSaving || isRunning}
            title={t("lab.saveConfig")}
          >
            {isSaving ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span className="text-[11px]">{t("lab.saving")}</span>
              </>
            ) : (
              <Save className="w-3.5 h-3.5" />
            )}
          </Button>
        )}

        {/* Run / Stop buttons */}
        {cell.type !== "display" && (
          isRunning && onStop ? (
            <Button variant="destructive" size="sm" className="h-7 px-2 gap-1" onClick={onStop} title="Stop execution">
              <Square className="w-3 h-3" />
            </Button>
          ) : onExecute ? (
            <Button
              variant={cell.status === "idle" ? "default" : "outline"}
              size="sm"
              className="h-7 px-2 gap-1 transition-all"
              onClick={onExecute}
              disabled={isExecuting || isSaving}
              title={cell.status === "idle" ? `Run ${cell.title || "cell"}` : "Re-run cell"}
            >
              {isExecuting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span className="text-[11px]">{t("lab.running")}</span>
                </>
              ) : (
                <Play className="w-3.5 h-3.5" />
              )}
            </Button>
          ) : null
        )}

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
              <MoreHorizontal className="w-3.5 h-3.5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {!isFirst && onMoveUp && (
              <DropdownMenuItem onClick={onMoveUp}>
                <ArrowUp className="w-3.5 h-3.5 mr-2" /> {t("lab.moveUp")}
              </DropdownMenuItem>
            )}
            {!isLast && onMoveDown && (
              <DropdownMenuItem onClick={onMoveDown}>
                <ArrowDown className="w-3.5 h-3.5 mr-2" /> {t("lab.moveDown")}
              </DropdownMenuItem>
            )}
            <DropdownMenuItem onClick={onDelete} className="text-destructive">
              <Trash2 className="w-3.5 h-3.5 mr-2" /> {t("lab.deleteCell")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Editor — always visible, takes at least 30% */}
      <div className="flex-1 min-h-[30%] overflow-y-auto px-4 py-3">
        {children}
      </div>

      {/* Combined panel: Output + Terminal tabs — always visible */}
      <CombinedPanel
        cell={cell}
        terminalLines={terminalLines}
        isStreaming={isStreaming}
        isRunning={isRunning}
        terminalRef={terminalRef}
        t={t}
      />
    </div>
  );
}

interface CombinedPanelProps {
  cell: Cell;
  terminalLines?: TerminalLine[];
  isStreaming?: boolean;
  isRunning: boolean;
  terminalRef: React.RefObject<HTMLDivElement | null>;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

function CombinedPanel({ cell, terminalLines, isStreaming, isRunning, terminalRef, t }: CombinedPanelProps) {
  const hasLogs = cell.type === "llm" && terminalLines && terminalLines.length > 0;
  const [tab, setTab] = useState<"output" | "logs">(hasLogs && isStreaming ? "logs" : "output");

  return (
    <div className="border-t flex flex-col min-h-[20%] max-h-[35%]">
      {/* Tab bar */}
      <div className="flex items-center border-b shrink-0 bg-card/50">
        <button
          onClick={() => setTab("output")}
          className={`px-4 py-1.5 text-xs font-medium border-b-2 transition-colors ${
            tab === "output"
              ? "border-primary text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          {t("lab.output")}
        </button>
        {hasLogs && (
          <button
            onClick={() => setTab("logs")}
            className={`px-4 py-1.5 text-xs font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
              tab === "logs"
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {t("lab.logs")}
            {isStreaming && <span className="w-1.5 h-1.5 rounded-full bg-[#3fb950] animate-pulse" />}
          </button>
        )}
      </div>

      {/* Tab content */}
      {tab === "output" && (
        <div className="overflow-y-auto flex-1">
          {isRunning && (
            <div className="flex items-center gap-2 px-4 py-2 text-sm text-amber-700 bg-amber-50 border-b border-amber-100">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>
                {cell.type === "llm" && "LLM inference in progress…"}
                {cell.type === "code" && "Code execution in progress…"}
                {!["llm", "code"].includes(cell.type) && `${t("lab.running")}…`}
              </span>
            </div>
          )}
          <div className="px-4 py-3">
            <CellOutput cell={cell} />
          </div>
        </div>
      )}

      {tab === "logs" && hasLogs && (
        <div ref={terminalRef} className="flex-1 overflow-y-auto bg-[#0d1117] px-4 py-2 font-mono text-[13px] leading-relaxed">
          {terminalLines.map((line, i) => (
            <div key={i} className={cn("whitespace-pre-wrap break-words", terminalLineColor(line.type))}>
              {line.content}
            </div>
          ))}
          {isStreaming && <span className="inline-block w-2 h-4 bg-[#58a6ff] animate-pulse" />}
        </div>
      )}
    </div>
  );
}

function terminalLineColor(type: TerminalLine["type"]): string {
  switch (type) {
    case "thinking":
      return "text-[#8b949e] italic";
    case "tool":
      return "text-[#d2a8ff]";
    case "status":
      return "text-[#58a6ff]";
    case "error":
      return "text-[#f85149]";
    default:
      return "text-[#c9d1d9]";
  }
}
