import { cn } from "@/lib/utils";
import type { Cell } from "@/hooks/use-lab";

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-emerald-500",
  running: "bg-amber-500",
  error: "bg-red-500",
  idle: "bg-slate-200 dark:bg-slate-700",
};

interface LabProgressProps {
  cells: Cell[];
  currentIndex: number;
  onSelect: (index: number) => void;
}

export function LabProgress({ cells, currentIndex, onSelect }: LabProgressProps) {
  if (cells.length === 0) return null;

  const completed = cells.filter((c) => c.status === "completed").length;
  const running = cells.filter((c) => c.status === "running").length;
  const errors = cells.filter((c) => c.status === "error").length;

  return (
    <div className="flex items-center gap-2 px-4 py-1.5 border-b bg-card/50 shrink-0">
      <div className="flex items-center gap-0.5 flex-1">
        {cells.map((cell, i) => (
          <button
            key={cell.id}
            onClick={() => onSelect(i)}
            className={cn(
              "h-1.5 flex-1 rounded-full transition-all cursor-pointer min-w-[6px]",
              STATUS_COLORS[cell.status] || STATUS_COLORS.idle,
              i === currentIndex && "h-2 ring-2 ring-primary/30",
            )}
            title={`${cell.title}: ${cell.status}`}
          />
        ))}
      </div>
      <span className="text-[11px] text-muted-foreground whitespace-nowrap tabular-nums min-w-[60px] text-right">
        {completed}/{cells.length}
        {running > 0 && ` · ${running} running`}
        {errors > 0 && ` · ${errors} err`}
      </span>
    </div>
  );
}
