import { useState } from "react";
import { useNavigate } from "react-router";
import { Plus, Trash2, FlaskConical, Pencil } from "lucide-react";
import { useLabs, useCreateLab, useDeleteLab } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/confirm-dialog";

function timeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMin = Math.floor((now - then) / 60000);
  if (diffMin < 1) return "Just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

export default function LabListPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { data: labs, isLoading } = useLabs();
  const createLab = useCreateLab();
  const deleteLab = useDeleteLab();

  const [showCreate, setShowCreate] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; title: string } | null>(null);

  const handleCreate = async () => {
    if (!newTitle.trim()) return;
    const lab = await createLab.mutateAsync({ title: newTitle.trim(), description: newDesc.trim() });
    setShowCreate(false);
    setNewTitle("");
    setNewDesc("");
    navigate(`/admin/labs/${lab.id}`);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">{t("labs.title")}</h1>
        <Button onClick={() => setShowCreate(true)}>
          <Plus className="w-4 h-4 mr-1" />
          {t("labs.newLab")}
        </Button>
      </div>

      {isLoading && <p className="text-sm text-muted-foreground">{t("common.loading")}</p>}

      {!isLoading && labs && labs.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <FlaskConical className="w-12 h-12 text-muted-foreground/50 mb-4" />
          <h2 className="text-lg font-medium">{t("labs.noLabs")}</h2>
          <p className="text-sm text-muted-foreground mt-1 max-w-sm">{t("labs.noLabsDesc")}</p>
          <Button className="mt-4" onClick={() => setShowCreate(true)}>
            <Plus className="w-4 h-4 mr-1" />
            {t("labs.createLab")}
          </Button>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {(labs || []).map((lab) => (
          <Card key={lab.id} className="p-4 hover:shadow-md transition-shadow cursor-pointer group" onClick={() => navigate(`/admin/labs/${lab.id}`)}>
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <h3 className="font-medium truncate">{lab.title}</h3>
                <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                  {lab.description || t("common.noDescription")}
                </p>
              </div>
            </div>
            <div className="flex items-center justify-between mt-3 pt-3 border-t">
              <span className="text-xs text-muted-foreground">{timeAgo(lab.updated_at)}</span>
              <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  className="p-1 hover:bg-muted rounded"
                  onClick={(e) => {
                    e.stopPropagation();
                    navigate(`/admin/labs/${lab.id}`);
                  }}
                >
                  <Pencil className="w-3.5 h-3.5" />
                </button>
                <button
                  className="p-1 hover:bg-destructive/10 rounded text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteTarget({ id: lab.id, title: lab.title });
                  }}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("labs.createTitle")}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">{t("labs.titleLabel")}</label>
              <Input
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                placeholder={t("labs.titlePlaceholder")}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                autoFocus
              />
            </div>
            <div>
              <label className="text-sm font-medium">{t("labs.descLabel")}</label>
              <Textarea
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder={t("labs.descPlaceholder")}
                rows={3}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowCreate(false)}>{t("common.cancel")}</Button>
              <Button onClick={handleCreate} disabled={!newTitle.trim() || createLab.isPending}>
                {createLab.isPending ? t("labs.savingLab") : t("common.save")}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={!!deleteTarget}
        onCancel={() => setDeleteTarget(null)}
        title={t("labs.deleteTitle")}
        description={t("labs.deleteDesc", { name: deleteTarget?.title || "" })}
        onConfirm={() => {
          if (deleteTarget) {
            deleteLab.mutate(deleteTarget.id);
            setDeleteTarget(null);
          }
        }}
      />
    </div>
  );
}
