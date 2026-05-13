import { useState } from "react";
import { PageLayout } from "@/components/layout/PageLayout";
import { useKeys, useCreateKey, useDeleteKey } from "@/hooks/use-api";
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Trash2 } from "lucide-react";
import { useTranslation } from "@/i18n";

export default function KeysList() {
  const { t } = useTranslation();
  const { data: keys, isLoading } = useKeys();
  const createKey = useCreateKey();
  const deleteKey = useDeleteKey();
  const [showAdd, setShowAdd] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", value: "", description: "" });

  const handleAdd = () => {
    if (!form.name.trim() || !form.value.trim()) return;
    createKey.mutate(form, {
      onSuccess: () => {
        setShowAdd(false);
        setForm({ name: "", value: "", description: "" });
      },
    });
  };

  const handleDelete = () => {
    if (!deleteTarget) return;
    deleteKey.mutate(deleteTarget, {
      onSuccess: () => setDeleteTarget(null),
    });
  };

  return (
    <PageLayout
      title={t("keys.title")}
      actions={
        <Button size="sm" onClick={() => setShowAdd(true)}>
          <Plus className="w-4 h-4 mr-1.5" />
          {t("keys.addKey")}
        </Button>
      }
    >
      {isLoading ? (
        <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
      ) : !keys || keys.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-sm text-muted-foreground">
            {t("keys.noKeys")}
          </p>
        </div>
      ) : (
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("keys.name")}</TableHead>
                <TableHead>{t("keys.description")}</TableHead>
                <TableHead>{t("keys.created")}</TableHead>
                <TableHead className="w-[60px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {keys.map((k: { name: string; description: string; created_at: string }) => (
                <TableRow key={k.name}>
                  <TableCell className="font-mono text-sm">{k.name}</TableCell>
                  <TableCell className="text-muted-foreground text-sm">{k.description || "—"}</TableCell>
                  <TableCell className="text-muted-foreground text-sm">
                    {new Date(k.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <button
                      onClick={() => setDeleteTarget(k.name)}
                      className="p-1.5 rounded-md hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      <Dialog open={showAdd} onOpenChange={setShowAdd}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("keys.addTitle")}</DialogTitle>
            <DialogDescription>
              {t("keys.addDesc")}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div>
              <Label htmlFor="key-name">{t("keys.nameLabel")}</Label>
              <Input
                id="key-name"
                placeholder={t("keys.namePlaceholder")}
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="key-value">{t("keys.valueLabel")}</Label>
              <Input
                id="key-value"
                type="password"
                placeholder={t("keys.valuePlaceholder")}
                value={form.value}
                onChange={(e) => setForm({ ...form, value: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="key-desc">{t("keys.descLabel")}</Label>
              <Input
                id="key-desc"
                placeholder={t("keys.descPlaceholder")}
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAdd(false)}>{t("common.cancel")}</Button>
            <Button onClick={handleAdd} disabled={!form.name.trim() || !form.value.trim()}>
              {t("keys.saveKey")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={!!deleteTarget}
        title={t("keys.deleteTitle")}
        description={t("keys.deleteDesc", { name: deleteTarget || "" })}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
        destructive={true}
      />
    </PageLayout>
  );
}
