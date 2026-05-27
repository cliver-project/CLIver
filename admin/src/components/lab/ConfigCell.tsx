import { useState, useEffect, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Cell } from "@/hooks/use-lab";

interface ConfigCellProps {
  cell: Cell;
  onSave: (outputs: Record<string, unknown>) => void;
}

interface FieldSchema {
  type: "text" | "select" | "checkbox";
  label?: string;
  required?: boolean;
  default?: unknown;
  options?: string[];
  placeholder?: string;
}

export function ConfigCell({ cell, onSave }: ConfigCellProps) {
  const schema = (cell.inputs.schema || {}) as Record<string, FieldSchema>;
  const [values, setValues] = useState<Record<string, unknown>>(() => {
    if (cell.outputs && Object.keys(cell.outputs).length > 0) {
      return { ...cell.outputs };
    }
    const defaults: Record<string, unknown> = {};
    for (const [key, field] of Object.entries(schema)) {
      if (field.default !== undefined) {
        defaults[key] = field.default;
      }
    }
    return defaults;
  });

  // Auto-save defaults on first mount if outputs is empty
  const didAutoSave = useRef(false);
  useEffect(() => {
    if (!didAutoSave.current && (!cell.outputs || Object.keys(cell.outputs).length === 0)) {
      didAutoSave.current = true;
      onSave(values);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-3">
      {Object.entries(schema).map(([key, field]) => (
        <div key={key}>
          <Label className="text-xs font-medium">
            {field.label || key}
            {field.required && <span className="text-red-500 ml-0.5">*</span>}
          </Label>

          {field.type === "select" && field.options ? (
            <Select
              value={String(values[key] || "")}
              onValueChange={(v) => { setValues({ ...values, [key]: v }); onSave({ ...values, [key]: v }); }}
            >
              <SelectTrigger className="mt-1">
                <SelectValue placeholder={`Select ${field.label || key}`} />
              </SelectTrigger>
              <SelectContent>
                {field.options.map((opt) => (
                  <SelectItem key={opt} value={opt}>{opt}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : field.type === "checkbox" ? (
            <div className="mt-1">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={Boolean(values[key])}
                  onChange={(e) => { setValues({ ...values, [key]: e.target.checked }); onSave({ ...values, [key]: e.target.checked }); }}
                  className="rounded border-border"
                />
                {field.label || key}
              </label>
            </div>
          ) : (
            <Input
              type="text"
              value={String(values[key] || "")}
              onChange={(e) => { setValues({ ...values, [key]: e.target.value }); }}
              onBlur={() => onSave(values)}
              placeholder={field.placeholder || `Enter ${field.label || key}`}
              className="mt-1"
            />
          )}
        </div>
      ))}
    </div>
  );
}
