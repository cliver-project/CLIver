import { Link, useNavigate } from "react-router";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useCreateTask } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import TaskForm from "./form";

export default function TaskCreatePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const createTask = useCreateTask();

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link to="/admin/tasks">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-4 h-4" />
          </Button>
        </Link>
        <h1 className="text-2xl font-bold">{t("tasks.createTask")}</h1>
      </div>

      <TaskForm
        mode="create"
        onSubmit={(data) => {
          createTask.mutate(data, {
            onSuccess: (result) => {
              const name = (result as Record<string, unknown>)?.name;
              navigate(name ? `/admin/tasks/${encodeURIComponent(String(name))}` : "/admin/tasks");
            },
          });
        }}
        isPending={createTask.isPending}
        error={createTask.isError ? t("tasks.createError", { error: createTask.error?.message ?? "Unknown" }) : undefined}
      />
    </div>
  );
}
