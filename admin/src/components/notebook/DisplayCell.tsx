import { MarkdownView } from "@/components/markdown-view";
import type { Cell } from "@/hooks/use-notebook";
import { useTranslation } from "@/i18n";

interface DisplayCellProps {
  cell: Cell;
}

export function DisplayCell({ cell }: DisplayCellProps) {
  const { t } = useTranslation();
  const content = (cell.inputs.content as string) || "";
  const format = (cell.inputs.format as string) || "markdown";

  if (!content) {
    return (
      <div className="text-sm text-muted-foreground italic">
        {t("notebook.noContent")}
      </div>
    );
  }

  // WARNING: HTML format should only be used with trusted content
  // Consider adding DOMPurify for production use to prevent XSS
  if (format === "html") {
    return (
      <div
        className="prose prose-sm max-w-none"
        dangerouslySetInnerHTML={{ __html: content }}
      />
    );
  }

  return <MarkdownView content={content} />;
}
