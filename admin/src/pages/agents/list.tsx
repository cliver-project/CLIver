import { useState } from "react";
import { Link } from "react-router";
import { Plus, Trash2, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useAgents, useConfig, useSaveConfig } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

export default function AgentListPage() {
  const { t } = useTranslation();
  const { data, isLoading } = useAgents();
  const { data: configData } = useConfig();
  const saveConfig = useSaveConfig();
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const agents = (data ?? []) as Array<Record<string, unknown>>;

  if (isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;

  const handleDelete = (name: string) => {
    if (!configData) return;
    const config = configData as Record<string, unknown>;
    const currentAgents = (config.agents ?? {}) as Record<string, Record<string, unknown>>;
    const updated = { ...currentAgents };
    delete updated[name];
    saveConfig.mutate(
      { ...config, agents: updated },
      { onSuccess: () => setDeleteTarget(null) },
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("agents.title")}</h1>
        <Link to="/admin/agents/new">
          <Button size="sm">
            <Plus className="w-4 h-4 mr-1" /> {t("agents.createAgent")}
          </Button>
        </Link>
      </div>
      {agents.length === 0 ? (
        <p className="text-muted-foreground">{t("agents.noAgents")}</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("agents.name")}</TableHead>
              <TableHead>{t("agents.description")}</TableHead>
              <TableHead>{t("agents.model")}</TableHead>
              <TableHead>{t("agents.skills")}</TableHead>
              <TableHead>{t("agents.isDefault")}</TableHead>
              <TableHead className="w-16" />
            </TableRow>
          </TableHeader>
          <TableBody className="[&>tr:nth-child(even)]:bg-muted/30">
            {agents.map((agent) => {
              const name = String(agent.name);
              const skills = (agent.skills ?? []) as string[];
              const isDefault = agent.is_default === true;
              return (
                <TableRow key={name}>
                  <TableCell>
                    <Link
                      to={`/admin/agents/${encodeURIComponent(name)}`}
                      className="text-primary hover:underline font-medium"
                    >
                      {name}
                    </Link>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm max-w-xs truncate">
                    {agent.description ? String(agent.description) : t("common.noDescription")}
                  </TableCell>
                  <TableCell>
                    {agent.model ? (
                      <Badge variant="secondary">{String(agent.model)}</Badge>
                    ) : (
                      <span className="text-muted-foreground text-xs">—</span>
                    )}
                  </TableCell>
                  <TableCell>
                    {skills.length > 0 ? (
                      <Badge variant="outline">{skills.length}</Badge>
                    ) : (
                      <span className="text-muted-foreground text-xs">—</span>
                    )}
                  </TableCell>
                  <TableCell>
                    {isDefault && (
                      <Badge variant="secondary" className="gap-1">
                        <Star className="w-3 h-3" />
                        {t("agents.defaultBadge")}
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => setDeleteTarget(name)}
                      title={t("common.delete")}
                    >
                      <Trash2 className="w-4 h-4 text-destructive" />
                    </Button>
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
