import { PageLayout } from "@/components/layout/PageLayout";
import { FolderOpen } from "lucide-react";

export default function ProjectsList() {
  return (
    <PageLayout title="Projects">
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="w-16 h-16 rounded-2xl bg-accent flex items-center justify-center mb-4">
          <FolderOpen className="w-8 h-8 text-primary" />
        </div>
        <h2 className="text-lg font-semibold text-foreground mb-2">Projects Coming Soon</h2>
        <p className="text-sm text-muted-foreground max-w-md">
          Organize your work into projects with issues and scenario templates.
          Each issue generates a notebook from a pre-configured scenario.
        </p>
      </div>
    </PageLayout>
  );
}
