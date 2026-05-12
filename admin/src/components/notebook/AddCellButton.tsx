import { useState } from "react";
import { Settings, Bot, Code2, FileText, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

const CELL_TYPES = [
  {
    type: "config",
    label: "Config",
    description: "Form inputs for configuration",
    icon: Settings,
    color: "text-indigo-600",
  },
  {
    type: "llm",
    label: "LLM",
    description: "AI agent prompt and response",
    icon: Bot,
    color: "text-purple-600",
  },
  {
    type: "code",
    label: "Code",
    description: "Python code execution",
    icon: Code2,
    color: "text-emerald-600",
  },
  {
    type: "display",
    label: "Display",
    description: "Markdown or HTML content",
    icon: FileText,
    color: "text-amber-600",
  },
] as const;

interface AddCellButtonProps {
  onAdd: (type: string) => void;
}

export function AddCellButton({ onAdd }: AddCellButtonProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      {!isOpen ? (
        <button
          onClick={() => setIsOpen(true)}
          className="w-full border border-dashed border-border rounded-lg py-3 text-sm text-muted-foreground hover:text-foreground hover:border-primary/50 transition-colors flex items-center justify-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Add Cell
        </button>
      ) : (
        <div className="border border-border rounded-lg p-2 bg-card shadow-sm">
          <div className="text-xs font-medium text-muted-foreground px-2 mb-1">Choose cell type</div>
          <div className="grid grid-cols-2 gap-1">
            {CELL_TYPES.map((ct) => (
              <button
                key={ct.type}
                onClick={() => {
                  onAdd(ct.type);
                  setIsOpen(false);
                }}
                className="flex items-start gap-2.5 p-2.5 rounded-md hover:bg-accent text-left transition-colors"
              >
                <ct.icon className={cn("w-4 h-4 mt-0.5 shrink-0", ct.color)} />
                <div>
                  <div className="text-sm font-medium">{ct.label}</div>
                  <div className="text-[11px] text-muted-foreground">{ct.description}</div>
                </div>
              </button>
            ))}
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="w-full mt-1 text-xs text-muted-foreground hover:text-foreground text-center py-1"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
