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

export default function KeysList() {
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
      title="Keys"
      actions={
        <Button size="sm" onClick={() => setShowAdd(true)}>
          <Plus className="w-4 h-4 mr-1.5" />
          Add Key
        </Button>
      }
    >
      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : !keys || keys.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-sm text-muted-foreground">
            No keys stored. Click "Add Key" to create one.
          </p>
        </div>
      ) : (
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Description</TableHead>
                <TableHead>Created</TableHead>
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
            <DialogTitle>Add Key</DialogTitle>
            <DialogDescription>
              Store an encrypted secret (API key, token, password).
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div>
              <Label htmlFor="key-name">Name</Label>
              <Input
                id="key-name"
                placeholder="e.g. openai_key"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="key-value">Value</Label>
              <Input
                id="key-value"
                type="password"
                placeholder="Enter secret value"
                value={form.value}
                onChange={(e) => setForm({ ...form, value: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="key-desc">Description (optional)</Label>
              <Input
                id="key-desc"
                placeholder="e.g. OpenAI API key for research"
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAdd(false)}>Cancel</Button>
            <Button onClick={handleAdd} disabled={!form.name.trim() || !form.value.trim()}>
              Save Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Key"
        description={`Are you sure you want to delete "${deleteTarget}"? This cannot be undone.`}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
        destructive={true}
      />
    </PageLayout>
  );
}
