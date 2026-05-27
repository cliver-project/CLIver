import { useState } from "react";
import { useNavigate } from "react-router";
import { PageLayout } from "@/components/layout/PageLayout";
import { useNotebooks, useCreateNotebook, useDeleteNotebook } from "@/hooks/use-notebook";
import { useScenarios } from "@/hooks/use-api";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CellStatusBadge } from "@/components/notebook/CellStatusBadge";
import { Plus, Book, Trash2 } from "lucide-react";
import type { NotebookSummary } from "@/hooks/use-notebook";
import { useTranslation } from "@/i18n";

export default function NotebooksList() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { data: notebooks, isLoading } = useNotebooks();
  const { data: scenarios } = useScenarios();
  const createNotebook = useCreateNotebook();
  const deleteNotebook = useDeleteNotebook();
  const [showCreate, setShowCreate] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [form, setForm] = useState({ title: "", description: "" });
  const [selectedScenario, setSelectedScenario] = useState<string>("");

  const handleCreate = async () => {
    if (!form.title.trim()) return;
    let cells: unknown[] = [];
    let defaultAgent: string | undefined;
    const scenarioId = selectedScenario && selectedScenario !== "__none__" ? selectedScenario : "";
    if (scenarioId) {
      try {
        const detail = await api<Record<string, unknown>>(`/scenarios/${encodeURIComponent(scenarioId)}`);
        const template = detail.template as Record<string, unknown> | undefined;
        if (template) {
          cells = (template.cells as unknown[]) || [];
          defaultAgent = template.default_agent as string;
        }
      } catch {
        /* ignore */
      }
    }
    createNotebook.mutate(
      {
        title: form.title,
        description: form.description,
        scenario_id: scenarioId || undefined,
        cells: cells as any,
        default_agent: defaultAgent,
      },
      {
        onSuccess: (nb) => {
          setShowCreate(false);
          setForm({ title: "", description: "" });
          setSelectedScenario("");
          navigate(`/admin/notebooks/${nb.id}`);
        },
      },
    );
  };

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteNotebook.mutate(deleteTarget, {
      onSuccess: () => setDeleteTarget(null),
    });
  };

  return (
    <PageLayout
      title={t("notebooks.title")}
      actions={
        <Button size="sm" onClick={() => setShowCreate(true)}>
          <Plus className="w-4 h-4 mr-1.5" />
          {t("notebooks.newNotebook")}
        </Button>
      }
    >
      {isLoading ? (
        <div className="text-sm text-muted-foreground">{t("common.loading")}</div>
      ) : !notebooks || notebooks.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="w-16 h-16 rounded-2xl bg-accent flex items-center justify-center mb-4">
            <Book className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-lg font-semibold text-foreground mb-2">{t("notebooks.noNotebooks")}</h2>
          <p className="text-sm text-muted-foreground max-w-md mb-4">
            {t("notebooks.noNotebooksDesc")}
          </p>
          <Button onClick={() => setShowCreate(true)}>
            <Plus className="w-4 h-4 mr-1.5" />
            {t("notebooks.createNotebook")}
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {notebooks.map((nb: NotebookSummary) => (
            <Card
              key={nb.id}
              className="cursor-pointer hover:shadow-md transition-shadow group"
              onClick={() => navigate(`/admin/notebooks/${nb.id}`)}
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <CardTitle className="text-sm font-medium">{nb.title}</CardTitle>
                  <div className="flex items-center gap-1">
                    <CellStatusBadge status={nb.status} />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteTarget(nb.id);
                      }}
                      className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-all"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {nb.description && (
                  <p className="text-xs text-muted-foreground mb-2 line-clamp-2">{nb.description}</p>
                )}
                <div className="text-[11px] text-muted-foreground">
                  {t("notebooks.cells", { count: nb.cell_count })} · {t("notebooks.updated", { date: new Date(nb.updated_at).toLocaleDateString() })}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Create Dialog */}
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("notebooks.createTitle")}</DialogTitle>
            <DialogDescription>{t("notebooks.createDesc")}</DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div>
              <Label htmlFor="nb-title">{t("notebooks.titleLabel")}</Label>
              <Input
                id="nb-title"
                placeholder={t("notebooks.titlePlaceholder")}
                value={form.title}
                onChange={(e) => setForm({ ...form, title: e.target.value })}
              />
            </div>
            <div>
              <Label>Scenario Template (optional)</Label>
              <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                <SelectTrigger>
                  <SelectValue placeholder="None — blank notebook" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None — blank notebook</SelectItem>
                  {scenarios && (scenarios as Array<Record<string, unknown>>).map((s) => (
                    <SelectItem key={String(s.id)} value={String(s.id)}>
                      {String(s.display_name)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="nb-desc">{t("notebooks.descLabel")}</Label>
              <Input
                id="nb-desc"
                placeholder={t("notebooks.descPlaceholder")}
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreate(false)}>{t("common.cancel")}</Button>
            <Button onClick={handleCreate} disabled={!form.title.trim()}>{t("notebooks.createNotebook")}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        title={t("notebooks.deleteTitle")}
        description={t("notebooks.deleteDesc")}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
        destructive={true}
      />
    </PageLayout>
  );
}
