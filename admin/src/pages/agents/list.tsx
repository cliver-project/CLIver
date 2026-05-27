import { useState } from "react";
import { Link } from "react-router";
import { Plus, Trash2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useAgents, useModels, useCreateAgent, useDeleteAgent, type AgentInfo } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

export default function AgentListPage() {
  const { t } = useTranslation();
  const { data, isLoading } = useAgents();
  const deleteAgent = useDeleteAgent();
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);

  const agents = (data ?? []) as AgentInfo[];

  if (isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;

  const handleDelete = (id: string) => {
    deleteAgent.mutate(id, { onSuccess: () => setDeleteTarget(null) });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("agents.title")}</h1>
        <Button size="sm" onClick={() => setShowCreate(true)}>
          <Plus className="w-4 h-4 mr-1" /> {t("agents.createAgent")}
        </Button>
      </div>

      {showCreate && (
        <CreateAgentDialog
          onSave={() => setShowCreate(false)}
          onCancel={() => setShowCreate(false)}
        />
      )}
      {agents.length === 0 ? (
        <p className="text-muted-foreground">{t("agents.noAgents")}</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("agents.name")}</TableHead>
              <TableHead>{t("agents.type")}</TableHead>
              <TableHead>{t("agents.description")}</TableHead>
              <TableHead>{t("agents.model")}</TableHead>
              <TableHead className="text-center">{t("agents.isDefault")}</TableHead>
              <TableHead className="w-16" />
            </TableRow>
          </TableHeader>
          <TableBody className="[&>tr:nth-child(even)]:bg-muted/30">
            {agents.map((agent) => {
              const isDefault = agent.is_default === 1;
              return (
                <TableRow key={agent.id}>
                  <TableCell>
                    <Link
                      to={`/admin/agents/${agent.id}`}
                      className="text-primary hover:underline font-medium"
                    >
                      {agent.name}
                    </Link>
                  </TableCell>
                  <TableCell className="text-sm">
                    {agent.type || "cliver"}
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm max-w-xs truncate">
                    {agent.description ? agent.description : t("common.noDescription")}
                  </TableCell>
                  <TableCell>
                    {agent.model ? (
                      <Badge variant="secondary">{agent.model}</Badge>
                    ) : (
                      <span className="text-muted-foreground text-xs">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-center">
                    {isDefault && <span className="text-emerald-600 text-base">✓</span>}
                  </TableCell>
                  <TableCell>
                    {!isDefault && (
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => setDeleteTarget(agent.id)}
                        title={t("common.delete")}
                      >
                        <Trash2 className="w-4 h-4 text-destructive" />
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
      {deleteTarget && (
        <ConfirmDialog
          open
          title={t("agents.deleteAgent")}
          description={t("agents.deleteAgentDescription", { name: deleteTarget })}
          destructive
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => handleDelete(deleteTarget)}
        />
      )}
    </div>
  );
}

function CreateAgentDialog({
  onSave,
  onCancel,
}: {
  onSave: () => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  const { data: models } = useModels();
  const createAgent = useCreateAgent();
  const [name, setName] = useState("");
  const [model, setModel] = useState("");
  const [desc, setDesc] = useState("");

  const handleSave = async () => {
    if (!name.trim()) return;
    await createAgent.mutateAsync({
      name: name.trim(),
      ...(model ? { model } : {}),
      ...(desc ? { description: desc } : {}),
    });
    onSave();
  };

  return (
    <Dialog open onOpenChange={() => onCancel()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t("agents.createAgent")}</DialogTitle>
          <DialogDescription>{t("agents.descriptionPlaceholder")}</DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <div>
            <Label>{t("agents.name")}</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. coder" />
          </div>
          <div>
            <Label>{t("agents.model")}</Label>
            <Select value={model || "__default__"} onValueChange={(v) => setModel(v === "__default__" ? "" : v)}>
              <SelectTrigger><SelectValue placeholder="(default)" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="__default__">(default)</SelectItem>
                {(models ?? []).map((m) => <SelectItem key={m.id} value={m.name}>{m.name}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>{t("agents.description")}</Label>
            <Input value={desc} onChange={(e) => setDesc(e.target.value)} placeholder={t("common.noDescription")} />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>{t("common.cancel")}</Button>
          <Button onClick={handleSave} disabled={!name.trim() || createAgent.isPending}>
            {createAgent.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin mr-1" />}
            {t("common.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
