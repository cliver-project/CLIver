import { useState } from "react";
import { Link2, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAvailableRefs, type RefGroup } from "@/hooks/use-notebook";
import { cn } from "@/lib/utils";

interface RefInsertDropdownProps {
  notebookId: string;
  cellId: string;
  onInsert: (ref: string) => void;
}

export function RefInsertDropdown({ notebookId, cellId, onInsert }: RefInsertDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const { data: refs, isLoading } = useAvailableRefs(notebookId, cellId);

  if (!isOpen) {
    return (
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsOpen(true)}
        className="h-7 text-xs text-muted-foreground"
        title="Insert reference to previous cell output"
      >
        <Link2 className="w-3.5 h-3.5 mr-1" />
        Insert Ref
      </Button>
    );
  }

  return (
    <div className="border border-border rounded-lg bg-card shadow-md p-2 min-w-[280px]">
      <div className="flex items-center justify-between mb-1.5 px-1">
        <span className="text-xs font-medium text-muted-foreground">Available References</span>
        <button
          onClick={() => setIsOpen(false)}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          Close
        </button>
      </div>

      {isLoading ? (
        <div className="text-xs text-muted-foreground p-2">Loading...</div>
      ) : !refs || refs.length === 0 ? (
        <div className="text-xs text-muted-foreground p-2">
          No references available. Execute previous cells first.
        </div>
      ) : (
        <div className="space-y-1 max-h-60 overflow-y-auto">
          {refs.map((group: RefGroup) => (
            <div key={group.cell_id}>
              <div className="text-[11px] font-medium text-muted-foreground px-1 py-0.5 flex items-center gap-1">
                <ChevronRight className="w-3 h-3" />
                {group.cell_title}
              </div>
              {group.fields.map((field) => (
                <button
                  key={field.path}
                  onClick={() => {
                    onInsert(`\${${field.path}}`);
                    setIsOpen(false);
                  }}
                  className="w-full text-left px-3 py-1.5 rounded hover:bg-accent text-xs transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-primary">{field.path.split(".").pop()}</span>
                    <span className={cn(
                      "text-[10px] px-1.5 py-0.5 rounded",
                      field.type === "string" && "bg-blue-50 text-blue-600",
                      field.type === "array" && "bg-purple-50 text-purple-600",
                      field.type === "object" && "bg-amber-50 text-amber-600",
                      field.type === "int" && "bg-emerald-50 text-emerald-600",
                    )}>
                      {field.type}
                    </span>
                  </div>
                  <div className="text-[10px] text-muted-foreground mt-0.5 truncate">
                    {field.preview}
                  </div>
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
