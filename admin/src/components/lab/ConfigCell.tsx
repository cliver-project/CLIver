import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Pencil } from "lucide-react";
import type { Cell } from "@/hooks/use-lab";
import { useTranslation } from "@/i18n";

interface ConfigCellProps {
  cell: Cell;
  onSave: (outputs: Record<string, unknown>) => void;
}

interface FieldSchema {
  type: string;
  label?: string;
  required?: boolean;
  default?: unknown;
  options?: string[];
  min?: number;
  max?: number;
  placeholder?: string;
}

export function ConfigCell({ cell, onSave }: ConfigCellProps) {
  const { t } = useTranslation();
  const schema = (cell.inputs.schema || {}) as Record<string, FieldSchema>;
  const [isEditing, setIsEditing] = useState(cell.status !== "completed");
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

  const handleSave = () => {
    onSave(values);
    setIsEditing(false);
  };

  if (!isEditing && cell.status === "completed") {
    return (
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{t("lab.configuration")}</div>
          <Button variant="ghost" size="sm" onClick={() => setIsEditing(true)} className="h-7 text-xs">
            <Pencil className="w-3 h-3 mr-1" />
            {t("lab.edit")}
          </Button>
        </div>
        <div className="rounded-md bg-muted/50 p-3 space-y-1">
          {Object.entries(cell.outputs).map(([key, value]) => (
            <div key={key} className="flex gap-2 text-sm">
              <span className="text-muted-foreground min-w-[120px]">{schema[key]?.label || key}:</span>
              <span className="font-medium text-foreground">{String(value)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

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
              onValueChange={(v) => setValues({ ...values, [key]: v })}
            >
              <SelectTrigger className="mt-1">
                <SelectValue placeholder={`Select ${field.label || key}`} />
              </SelectTrigger>
              <SelectContent>
                {field.options.map((opt) => (
                  <SelectItem key={opt} value={opt}>
                    {opt}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : field.type === "range" ? (
            <div className="flex items-center gap-2 mt-1">
              <Input
                type="number"
                min={field.min}
                max={field.max}
                value={String((values[key] as number[])?.[0] ?? field.min ?? 0)}
                onChange={(e) => {
                  const arr = (values[key] as number[]) || [field.min ?? 0, field.max ?? 100];
                  setValues({ ...values, [key]: [Number(e.target.value), arr[1]] });
                }}
                className="w-24"
              />
              <span className="text-muted-foreground text-sm">to</span>
              <Input
                type="number"
                min={field.min}
                max={field.max}
                value={String((values[key] as number[])?.[1] ?? field.max ?? 100)}
                onChange={(e) => {
                  const arr = (values[key] as number[]) || [field.min ?? 0, field.max ?? 100];
                  setValues({ ...values, [key]: [arr[0], Number(e.target.value)] });
                }}
                className="w-24"
              />
            </div>
          ) : field.type === "checkbox" ? (
            <div className="mt-1">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={Boolean(values[key])}
                  onChange={(e) => setValues({ ...values, [key]: e.target.checked })}
                  className="rounded border-border"
                />
                {field.label || key}
              </label>
            </div>
          ) : field.type === "password" ? (
            <Input
              type="password"
              value={String(values[key] || "")}
              onChange={(e) => setValues({ ...values, [key]: e.target.value })}
              placeholder={field.placeholder || `Enter ${field.label || key}`}
              className="mt-1"
            />
          ) : (
            <Input
              type="text"
              value={String(values[key] || "")}
              onChange={(e) => setValues({ ...values, [key]: e.target.value })}
              placeholder={field.placeholder || `Enter ${field.label || key}`}
              className="mt-1"
            />
          )}
        </div>
      ))}

      <div className="flex justify-end pt-1">
        <Button size="sm" onClick={handleSave}>
          {t("lab.saveConfig")}
        </Button>
      </div>
    </div>
  );
}
