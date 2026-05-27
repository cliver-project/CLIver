import { PageLayout } from "@/components/layout/PageLayout";
import { Book } from "lucide-react";

export default function NotebooksList() {
  return (
    <PageLayout title="Notebooks">
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="w-16 h-16 rounded-2xl bg-accent flex items-center justify-center mb-4">
          <Book className="w-8 h-8 text-primary" />
        </div>
        <h2 className="text-lg font-semibold text-foreground mb-2">Notebooks Coming Soon</h2>
        <p className="text-sm text-muted-foreground max-w-md">
          Interactive AI notebooks with wizard-style cells for guided workflows.
          Create research papers, picture books, videos, and more.
        </p>
      </div>
    </PageLayout>
  );
}
