import { Badge } from "@/components/ui/badge";
import { useTranslation } from "@/i18n";
import type { TestRunResult } from "@/hooks/use-api";

interface GoldenTestCardProps {
  name: string;
  input: string;
  expectedOutput: string;
  result?: TestRunResult;
}

export function GoldenTestCard({ name, input, expectedOutput, result }: GoldenTestCardProps) {
  const { t } = useTranslation();

  return (
    <div className="rounded-md border p-2 text-[11px] space-y-1">
      <div className="flex items-center gap-2">
        <span className="font-medium">{name}</span>
        {result ? (
          <Badge variant="secondary" className="text-[10px]">{t("lab.passing")}</Badge>
        ) : (
          <Badge variant="outline" className="text-[10px] text-muted-foreground">{t("lab.notRun")}</Badge>
        )}
      </div>
      <p className="text-muted-foreground">
        <strong>{t("lab.testInput")}:</strong> {input.slice(0, 80)}{input.length > 80 ? "..." : ""}
      </p>
      <p className="text-muted-foreground">
        <strong>{t("lab.expectedOutput")}:</strong> {expectedOutput.slice(0, 80)}{expectedOutput.length > 80 ? "..." : ""}
      </p>
    </div>
  );
}
