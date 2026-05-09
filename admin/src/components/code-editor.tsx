import { useCallback, type KeyboardEvent } from "react";
import { cn } from "@/lib/utils";

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  language?: string;
  className?: string;
  minHeight?: string;
}

export function CodeEditor({
  value,
  onChange,
  readOnly = false,
  language,
  className,
  minHeight = "400px",
}: CodeEditorProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Tab" && !readOnly) {
        e.preventDefault();
        const target = e.currentTarget;
        const start = target.selectionStart;
        const end = target.selectionEnd;
        const newValue = value.substring(0, start) + "  " + value.substring(end);
        onChange(newValue);
        requestAnimationFrame(() => {
          target.selectionStart = target.selectionEnd = start + 2;
        });
      }
    },
    [value, onChange, readOnly],
  );

  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={handleKeyDown}
      readOnly={readOnly}
      spellCheck={false}
      data-language={language}
      className={cn(
        "w-full rounded-md border border-input bg-background px-3 py-2",
        "font-mono text-sm leading-relaxed resize-y",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        readOnly && "bg-muted cursor-default opacity-80",
        className,
      )}
      style={{ minHeight }}
    />
  );
}
