import { Settings, Bot, Code2, FileText, ChevronRight, ChevronDown, Play, MoreHorizontal, Trash2, ArrowUp, ArrowDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { CellStatusBadge } from "@/components/notebook/CellStatusBadge";
import { cn } from "@/lib/utils";
import type { Cell } from "@/hooks/use-notebook";
import { useTranslation } from "@/i18n";

const TYPE_CONFIG: Record<string, { icon: React.ComponentType<{ className?: string }>; color: string; borderColor: string; bgColor: string; labelKey: string }> = {
  config: { icon: Settings, color: "text-indigo-600", borderColor: "border-l-indigo-500", bgColor: "bg-indigo-50", labelKey: "notebook.config" },
  llm: { icon: Bot, color: "text-purple-600", borderColor: "border-l-purple-500", bgColor: "bg-purple-50", labelKey: "notebook.llm" },
  code: { icon: Code2, color: "text-emerald-600", borderColor: "border-l-emerald-500", bgColor: "bg-emerald-50", labelKey: "notebook.code" },
  display: { icon: FileText, color: "text-amber-600", borderColor: "border-l-amber-500", bgColor: "bg-amber-50", labelKey: "notebook.display" },
};

interface CellCardProps {
  cell: Cell;
  isExpanded: boolean;
  onToggle: () => void;
  onExecute: () => void;
  onDelete: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  isFirst?: boolean;
  isLast?: boolean;
  children: React.ReactNode;
}

export function CellCard({
  cell,
  isExpanded,
  onToggle,
  onExecute,
  onDelete,
  onMoveUp,
  onMoveDown,
  isFirst,
  isLast,
  children,
}: CellCardProps) {
  const { t } = useTranslation();
  const config = TYPE_CONFIG[cell.type] || TYPE_CONFIG["display"]!;
  const Icon = config.icon;

  const summary = getSummary(cell);

  return (
    <div
      className={cn(
        "bg-card border rounded-lg border-l-[3px] transition-shadow",
        config.borderColor,
        isExpanded ? "shadow-sm border-primary/30" : "hover:shadow-sm",
      )}
    >
      {/* Header — always visible, click to toggle */}
      <div
        className={cn(
          "flex items-center gap-2 px-3 py-2.5 cursor-pointer select-none",
          !isExpanded && "hover:bg-accent/30",
        )}
        onClick={onToggle}
      >
        {/* Expand/collapse arrow */}
        {isExpanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
        )}

        {/* Type icon */}
        <div className={cn("w-5 h-5 rounded flex items-center justify-center shrink-0", config.bgColor)}>
          <Icon className={cn("w-3 h-3", config.color)} />
        </div>

        {/* Title */}
        <span className="text-sm font-medium text-foreground truncate flex-1">
          {cell.title || `${t(config.labelKey)} Cell`}
        </span>

        {/* Status */}
        <CellStatusBadge status={cell.status} />

        {/* Actions (only when expanded) */}
        {isExpanded && (
          <div className="flex items-center gap-1 ml-1" onClick={(e) => e.stopPropagation()}>
            {cell.type !== "display" && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={onExecute}
                disabled={cell.status === "running"}
                title="Run cell"
              >
                <Play className="w-3.5 h-3.5" />
              </Button>
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
                    <ArrowUp className="w-3.5 h-3.5 mr-2" /> {t("notebook.moveUp")}
                  </DropdownMenuItem>
                )}
                {!isLast && onMoveDown && (
                  <DropdownMenuItem onClick={onMoveDown}>
                    <ArrowDown className="w-3.5 h-3.5 mr-2" /> {t("notebook.moveDown")}
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem onClick={onDelete} className="text-destructive">
                  <Trash2 className="w-3.5 h-3.5 mr-2" /> {t("notebook.deleteCell")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </div>

      {/* Collapsed summary */}
      {!isExpanded && summary && (
        <div className="px-3 pb-2.5 pl-[52px]">
          <div className="text-xs text-muted-foreground truncate">{summary}</div>
        </div>
      )}

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-1 border-t border-border/50">
          {children}
        </div>
      )}
    </div>
  );
}

function getSummary(cell: Cell): string {
  if (cell.status === "completed") {
    if (cell.type === "config") {
      const entries = Object.entries(cell.outputs).slice(0, 3);
      return entries.map(([k, v]) => `${k}: ${String(v)}`).join(" · ");
    }
    if (cell.type === "llm") {
      const text = cell.outputs.text as string;
      return text ? text.substring(0, 100) + (text.length > 100 ? "..." : "") : "Completed";
    }
    if (cell.type === "code") {
      const keys = Object.keys(cell.outputs);
      return keys.length > 0 ? `Output: {${keys.join(", ")}}` : "Completed";
    }
  }
  if (cell.status === "error") {
    return cell.error || "Error occurred";
  }
  if (cell.type === "llm") {
    const prompt = cell.inputs.prompt as string;
    return prompt ? prompt.substring(0, 80) + (prompt.length > 80 ? "..." : "") : "";
  }
  return "";
}
