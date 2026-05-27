import type { ReactNode } from "react";

interface LabHeaderProps {
  title: string;
  description: string;
  breadcrumb?: ReactNode;
}

export function LabHeader({ title, description, breadcrumb }: LabHeaderProps) {
  return (
    <div className="shrink-0 border-b bg-card px-4 py-2">
      {breadcrumb ? (
        <div className="flex items-center gap-1 text-xs text-muted-foreground [&_button]:cursor-pointer">
          {breadcrumb}
          {description && (
            <span className="text-muted-foreground/70 truncate hidden sm:inline ml-1">— {description}</span>
          )}
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <h1 className="text-sm font-semibold truncate">{title}</h1>
          {description && (
            <span className="text-xs text-muted-foreground truncate hidden sm:inline">— {description}</span>
          )}
        </div>
      )}
    </div>
  );
}
