import ReactMarkdown from "react-markdown";

export function MarkdownView({ content, className }: { content: string; className?: string }) {
  return (
    <div className={`prose prose-sm dark:prose-invert max-w-none
      prose-headings:mt-4 prose-headings:mb-2 prose-headings:font-semibold
      prose-p:my-1.5 prose-li:my-0.5
      prose-pre:bg-muted prose-pre:text-foreground prose-pre:rounded-md
      prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs
      prose-table:text-sm prose-th:px-2 prose-th:py-1 prose-td:px-2 prose-td:py-1
      prose-a:text-primary ${className ?? ""}`}>
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}
