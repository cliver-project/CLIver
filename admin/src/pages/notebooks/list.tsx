import { useState } from "react";
import { useNavigate } from "react-router";
import { PageLayout } from "@/components/layout/PageLayout";
import { useNotebooks, useCreateNotebook, useDeleteNotebook } from "@/hooks/use-notebook";
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
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CellStatusBadge } from "@/components/notebook/CellStatusBadge";
import { Plus, Book, Trash2 } from "lucide-react";
import type { NotebookSummary } from "@/hooks/use-notebook";

export default function NotebooksList() {
  const navigate = useNavigate();
  const { data: notebooks, isLoading } = useNotebooks();
  const createNotebook = useCreateNotebook();
  const deleteNotebook = useDeleteNotebook();
  const [showCreate, setShowCreate] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [form, setForm] = useState({ title: "", description: "" });

  const handleCreate = () => {
    if (!form.title.trim()) return;
    createNotebook.mutate(
      { title: form.title, description: form.description },
      {
        onSuccess: (nb) => {
          setShowCreate(false);
          setForm({ title: "", description: "" });
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
      title="Notebooks"
      actions={
        <Button size="sm" onClick={() => setShowCreate(true)}>
          <Plus className="w-4 h-4 mr-1.5" />
          New Notebook
        </Button>
      }
    >
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading...</div>
      ) : !notebooks || notebooks.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="w-16 h-16 rounded-2xl bg-accent flex items-center justify-center mb-4">
            <Book className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-lg font-semibold text-foreground mb-2">No Notebooks Yet</h2>
          <p className="text-sm text-muted-foreground max-w-md mb-4">
            Create your first notebook to start building AI workflows.
          </p>
          <Button onClick={() => setShowCreate(true)}>
            <Plus className="w-4 h-4 mr-1.5" />
            Create Notebook
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
                  {nb.cell_count} cells · Updated {new Date(nb.updated_at).toLocaleDateString()}
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
            <DialogTitle>New Notebook</DialogTitle>
            <DialogDescription>Create a new notebook to start building an AI workflow.</DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div>
              <Label htmlFor="nb-title">Title</Label>
              <Input
                id="nb-title"
                placeholder="e.g. Research Paper Analysis"
                value={form.title}
                onChange={(e) => setForm({ ...form, title: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="nb-desc">Description (optional)</Label>
              <Input
                id="nb-desc"
                placeholder="Brief description of this notebook"
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreate(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!form.title.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Notebook"
        description="Are you sure? This will permanently delete the notebook and all its cells."
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
        destructive={true}
      />
    </PageLayout>
  );
}
