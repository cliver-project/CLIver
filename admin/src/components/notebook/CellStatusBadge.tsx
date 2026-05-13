import { cn } from "@/lib/utils";
import { useTranslation } from "@/i18n";

const STATUS_STYLES: Record<string, { bg: string; text: string; labelKey: string; pulse?: boolean }> = {
  idle: { bg: "bg-muted", text: "text-muted-foreground", labelKey: "notebook.idle" },
  running: { bg: "bg-amber-100", text: "text-amber-700", labelKey: "notebook.running", pulse: true },
  completed: { bg: "bg-emerald-100", text: "text-emerald-700", labelKey: "notebook.completed" },
  error: { bg: "bg-red-100", text: "text-red-700", labelKey: "notebook.error" },
};

interface CellStatusBadgeProps {
  status: string;
  className?: string;
}

export function CellStatusBadge({ status, className }: CellStatusBadgeProps) {
  const { t } = useTranslation();
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
      {t(style.labelKey)}
    </span>
  );
}
