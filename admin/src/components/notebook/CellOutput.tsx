import { MarkdownView } from "@/components/markdown-view";
import { FileDown, ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

interface CellOutputProps {
  outputs: Record<string, unknown>;
  error?: string | null;
  status: string;
}

export function CellOutput({ outputs, error, status }: CellOutputProps) {
  const [showJson, setShowJson] = useState(false);

  if (status === "error" && error) {
    return (
      <div className="mt-3 rounded-lg bg-red-50 border border-red-200 p-3">
        <div className="text-sm font-medium text-red-800 mb-1">Execution Error</div>
        <div className="text-sm text-red-700">{error}</div>
      </div>
    );
  }

  if (status !== "completed" || !outputs || Object.keys(outputs).length === 0) {
    return null;
  }

  const text = outputs.text as string | undefined;
  const data = outputs.data as unknown | undefined;
  const artifacts = outputs.artifacts as Array<{
    path: string;
    media_type: string;
    size?: number;
  }> | undefined;

  return (
    <div className="mt-3 space-y-2">
      {text && (
        <div className="rounded-lg bg-muted/50 p-3">
          <MarkdownView content={text} />
        </div>
      )}

      {data !== undefined && (
        <div>
          <button
            onClick={() => setShowJson(!showJson)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {showJson ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            Structured Data
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
          <div className="text-xs font-medium text-muted-foreground">Artifacts</div>
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
