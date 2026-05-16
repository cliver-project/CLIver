import { CodeEditor } from "@/components/code-editor";
import { CellOutput } from "@/components/lab/CellOutput";
import type { Cell } from "@/hooks/use-lab";
import { useTranslation } from "@/i18n";

const DEFAULT_SOURCE = 'def run(ctx):\n    # Access previous cell outputs: ctx.refs("cell_id.outputs.field")\n    \n    return {"result": "hello"}';

interface CodeCellProps {
  cell: Cell;
  onSourceChange: (source: string) => void;
}

export function CodeCell({ cell, onSourceChange }: CodeCellProps) {
  const { t } = useTranslation();
  const source = (cell.inputs.source as string) || DEFAULT_SOURCE;
  const isHidden = cell.inputs.hidden === true;

  if (isHidden) {
    return (
      <div className="text-xs text-muted-foreground italic">
        {t("lab.codeHidden")}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        {t("lab.pythonCode")}
      </div>
      <CodeEditor
        value={source}
        onChange={onSourceChange}
        language="python"
        readOnly={cell.status === "running"}
        minHeight="120px"
        className="text-sm"
      />
      <CellOutput outputs={cell.outputs} error={cell.error} status={cell.status} />
    </div>
  );
}
