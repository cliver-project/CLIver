import { cn } from "@/lib/utils";

const STATUS_STYLES: Record<string, { bg: string; text: string; label: string; pulse?: boolean }> = {
  idle: { bg: "bg-muted", text: "text-muted-foreground", label: "Idle" },
  running: { bg: "bg-amber-100", text: "text-amber-700", label: "Running...", pulse: true },
  completed: { bg: "bg-emerald-100", text: "text-emerald-700", label: "Completed" },
  error: { bg: "bg-red-100", text: "text-red-700", label: "Error" },
};

interface CellStatusBadgeProps {
  status: string;
  className?: string;
}

export function CellStatusBadge({ status, className }: CellStatusBadgeProps) {
  const style = STATUS_STYLES[status] || STATUS_STYLES["idle"]!;
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium",
        style.bg,
        style.text,
        style.pulse && "animate-pulse",
        className,
      )}
    >
      {style.label}
    </span>
  );
}
