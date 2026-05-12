import { useState } from "react";
import { Link } from "react-router";
import { Play, Trash2, Loader2, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { StatusPill } from "@/components/status-pill";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useTasks, useRunTask, useDeleteTask } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

export default function TaskListPage() {
  const { t } = useTranslation();
  const { data, isLoading } = useTasks();
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const tasks = (data ?? []) as Array<Record<string, unknown>>;

  if (isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("tasks.title")}</h1>
        <Link to="/admin/tasks/new">
          <Button size="sm">
            <Plus className="w-4 h-4 mr-1" /> {t("tasks.createTask")}
          </Button>
        </Link>
      </div>
      {tasks.length === 0 ? (
        <p className="text-muted-foreground">{t("tasks.noTasks")}</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("tasks.name")}</TableHead>
              <TableHead>{t("tasks.schedule")}</TableHead>
              <TableHead>{t("tasks.status")}</TableHead>
              <TableHead className="w-24">{t("tasks.actions")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody className="[&>tr:nth-child(even)]:bg-muted/30">
            {tasks.map((tk) => (
              <TaskRow key={String(tk.name)} task={tk} onDelete={() => setDeleteTarget(String(tk.name))} />
            ))}
          </TableBody>
        </Table>
      )}
      {deleteTarget && (
        <DeleteDialog name={deleteTarget} onClose={() => setDeleteTarget(null)} />
      )}
    </div>
  );
}

function TaskRow({ task, onDelete }: { task: Record<string, unknown>; onDelete: () => void }) {
  const { t } = useTranslation();
  const name = String(task.name);
  const runTask = useRunTask(name);

  return (
    <TableRow>
      <TableCell>
        <Link to={`/admin/tasks/${encodeURIComponent(name)}`} className="text-primary hover:underline font-medium">
          {name}
        </Link>
      </TableCell>
      <TableCell className="text-muted-foreground text-xs">
        {(() => {
          const si = task.schedule_info as Record<string, unknown> | undefined;
          if (!si) return String(task.schedule ?? "—");
          if (si.type === "one-shot")
            return <span title={String(si.run_at ?? "")}>{t("tasks.oneShot")}{si.time_until ? ` (${si.time_until})` : ""}</span>;
          if (si.type === "cron")
            return (
              <span title={String(si.expression ?? "")}>
                {String(si.summary ?? si.expression ?? "cron")}
                {si.time_until ? <span className="text-muted-foreground"> — next in {String(si.time_until)}</span> : ""}
              </span>
            );
          return t("tasks.manual");
        })()}
      </TableCell>
      <TableCell>
        <StatusPill status={String(task.task_status ?? task.status ?? "pending")} />
      </TableCell>
      <TableCell>
        <div className="flex gap-1">
          <Button
            size="icon"
            variant="ghost"
            onClick={() => runTask.mutate()}
            disabled={runTask.isPending}
            title={runTask.isPending ? t("tasks.starting") : t("common.run")}
          >
            {runTask.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
          </Button>
          <Button size="icon" variant="ghost" onClick={onDelete} title={t("common.delete")}>
            <Trash2 className="w-4 h-4 text-destructive" />
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
}

function DeleteDialog({ name, onClose }: { name: string; onClose: () => void }) {
  const { t } = useTranslation();
  const deleteTask = useDeleteTask(name);
  return (
    <ConfirmDialog
      open
      title={t("tasks.deleteTask")}
      description={t("tasks.deleteTaskDescription", { name })}
      destructive
      onCancel={onClose}
      onConfirm={() => { deleteTask.mutate(); onClose(); }}
    />
  );
}
