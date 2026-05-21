interface LabHeaderProps {
  title: string;
  description: string;
}

export function LabHeader({ title, description }: LabHeaderProps) {
  return (
    <div className="shrink-0 border-b bg-card px-4 py-3">
      <h1 className="text-sm font-semibold truncate">{title}</h1>
      {description && (
        <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">{description}</p>
      )}
    </div>
  );
}
