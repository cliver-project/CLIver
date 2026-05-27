import { MarkdownView } from "@/components/markdown-view";
import { FileDown, ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "@/i18n";
import type { Cell } from "@/hooks/use-lab";

interface CellOutputProps {
  cell: Cell;
}

export function CellOutput({ cell }: CellOutputProps) {
  const [showJson, setShowJson] = useState(false);
  const { t } = useTranslation();
  const { outputs, error, status } = cell;

  if (status === "error" && error) {
    return (
      <div className="rounded-lg bg-red-50 border border-red-200 p-3">
        <div className="text-sm font-medium text-red-800 mb-1">{t("lab.executionError")}</div>
        <div className="text-sm text-red-700">{error}</div>
      </div>
    );
  }

  if (status === "running") {
    return (
      <div className="rounded-lg bg-amber-50 border border-amber-200 p-3">
        <div className="text-sm text-amber-700 flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
          {t("lab.running")}
        </div>
      </div>
    );
  }

  if (status !== "completed" || !outputs || Object.keys(outputs).length === 0) {
    return null;
  }

  const text = outputs.text as string | undefined;
  const result = outputs.result as string | undefined;
  const response = outputs.response as string | undefined;
  const displayText = text || result || response;

  const data = outputs.data as unknown | undefined;
  const artifacts = outputs.artifacts as Array<{
    path: string;
    media_type: string;
    size?: number;
  }> | undefined;

  // For config cells, show all output keys as a table
  const isConfig = cell.type === "config";
  const otherKeys = Object.entries(outputs).filter(
    ([k]) => !["text", "result", "response", "data", "artifacts"].includes(k),
  );

  return (
    <div className="space-y-2">
      {displayText && (
        <div className="rounded-lg bg-muted/50 p-3">
          <MarkdownView content={displayText} />
        </div>
      )}

      {isConfig && otherKeys.length > 0 && (
        <div className="rounded-lg border p-3">
          <div className="text-xs font-medium text-muted-foreground mb-2">{t("lab.configuration")}</div>
          <div className="space-y-1.5">
            {otherKeys.map(([key, value]) => (
              <div key={key} className="flex items-baseline gap-2 text-sm">
                <code className="text-[11px] text-primary/70 bg-primary/5 px-1 rounded font-mono">
                  {cell.id}.outputs.{key}
                </code>
                <span className="text-muted-foreground">=</span>
                <span className="text-foreground">{formatOutputValue(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {data !== undefined && (
        <div>
          <button
            onClick={() => setShowJson(!showJson)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {showJson ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            {t("lab.structuredData")}
          </button>
          {showJson && (
            <pre className="mt-1 rounded-md bg-muted p-3 text-xs overflow-auto max-h-60">
              {JSON.stringify(data, null, 2)}
            </pre>
          )}
        </div>
      )}

      {artifacts && artifacts.length > 0 && (
        <div className="space-y-1">
          <div className="text-xs font-medium text-muted-foreground">{t("lab.artifacts")}</div>
          {artifacts.map((a, i) => (
            <div
              key={i}
              className="flex items-center gap-2 text-sm text-primary hover:underline cursor-pointer"
            >
              <FileDown className="w-3.5 h-3.5" />
              <span>{a.path.split("/").pop()}</span>
              <span className="text-xs text-muted-foreground">({a.media_type})</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatOutputValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.map(String).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
