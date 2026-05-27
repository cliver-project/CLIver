import { useState } from "react";
import { useNavigate } from "react-router";
import { PageLayout } from "@/components/layout/PageLayout";
import { useLabs, useCreateLab, useDeleteLab } from "@/hooks/use-lab";
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
import { CellStatusBadge } from "@/components/lab/CellStatusBadge";
import { Plus, Book, Trash2 } from "lucide-react";
import type { LabSummary } from "@/hooks/use-lab";
import { useTranslation } from "@/i18n";

export default function LabsList() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { data: labs, isLoading } = useLabs();
  const { data: scenarios } = useScenarios();
  const createLab = useCreateLab();
  const deleteLab = useDeleteLab();
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
    createLab.mutate(
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
          navigate(`/admin/labs/${nb.id}`);
        },
      },
    );
  };

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteLab.mutate(deleteTarget, {
      onSuccess: () => setDeleteTarget(null),
    });
  };

  return (
    <PageLayout
      title={t("labs.title")}
      actions={
        <Button size="sm" onClick={() => setShowCreate(true)}>
          <Plus className="w-4 h-4 mr-1.5" />
          {t("labs.newLab")}
        </Button>
      }
    >
      {isLoading ? (
        <div className="text-sm text-muted-foreground">{t("common.loading")}</div>
      ) : !labs || labs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="w-16 h-16 rounded-2xl bg-accent flex items-center justify-center mb-4">
            <Book className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-lg font-semibold text-foreground mb-2">{t("labs.noLabs")}</h2>
          <p className="text-sm text-muted-foreground max-w-md mb-4">
            {t("labs.noLabsDesc")}
          </p>
          <Button onClick={() => setShowCreate(true)}>
            <Plus className="w-4 h-4 mr-1.5" />
            {t("labs.createLab")}
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {labs.map((nb: LabSummary) => (
            <Card
              key={nb.id}
              className="cursor-pointer hover:shadow-md transition-shadow group"
              onClick={() => navigate(`/admin/labs/${nb.id}`)}
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
                  {t("labs.cells", { count: nb.cell_count })} · {t("labs.updated", { date: new Date(nb.updated_at).toLocaleDateString() })}
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
            <DialogTitle>{t("labs.createTitle")}</DialogTitle>
            <DialogDescription>{t("labs.createDesc")}</DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div>
              <Label htmlFor="nb-title">{t("labs.titleLabel")}</Label>
              <Input
                id="nb-title"
                placeholder={t("labs.titlePlaceholder")}
                value={form.title}
                onChange={(e) => setForm({ ...form, title: e.target.value })}
              />
            </div>
            <div>
              <Label>Scenario Template (optional)</Label>
              <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                <SelectTrigger>
                  <SelectValue placeholder="None — blank lab" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None — blank lab</SelectItem>
                  {scenarios && (scenarios as Array<Record<string, unknown>>).map((s) => (
                    <SelectItem key={String(s.id)} value={String(s.id)}>
                      {String(s.display_name)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="nb-desc">{t("labs.descLabel")}</Label>
              <Input
                id="nb-desc"
                placeholder={t("labs.descPlaceholder")}
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreate(false)}>{t("common.cancel")}</Button>
            <Button onClick={handleCreate} disabled={!form.title.trim()}>{t("labs.createLab")}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        title={t("labs.deleteTitle")}
        description={t("labs.deleteDesc")}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
        destructive={true}
      />
    </PageLayout>
  );
}
