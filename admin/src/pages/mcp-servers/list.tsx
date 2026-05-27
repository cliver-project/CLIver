import { useState } from "react";
import { Plus, Trash2, Server, Pencil } from "lucide-react";
import { useMCPServers, useCreateMCPServer, useUpdateMCPServer, useDeleteMCPServer, type MCPServer } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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

const TRANSPORT_LABELS: Record<string, string> = {
  stdio: "STDIO",
  sse: "SSE",
  streamable_http: "HTTP",
  websocket: "WS",
};

function ServerForm({ server, onChange, t }: {
  server: Partial<MCPServer>;
  onChange: (s: Partial<MCPServer>) => void;
  t: (key: string, params?: Record<string, string>) => string;
}) {
  const isStdio = server.transport === "stdio";
  const set = (k: string, v: unknown) => onChange({ ...server, [k]: v });

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm font-medium">{t("mcpServers.nameLabel")}</label>
        <Input
          value={server.name || ""}
          onChange={(e) => set("name", e.target.value)}
          placeholder={t("mcpServers.namePlaceholder")}
          autoFocus
        />
      </div>
      <div>
        <label className="text-sm font-medium">{t("mcpServers.transport")}</label>
        <select
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={server.transport || "stdio"}
          onChange={(e) => set("transport", e.target.value)}
        >
          <option value="stdio">stdio</option>
          <option value="sse">sse</option>
          <option value="streamable_http">streamable_http</option>
          <option value="websocket">websocket</option>
        </select>
      </div>
      {isStdio ? (
        <>
          <div>
            <label className="text-sm font-medium">{t("mcpServers.command")}</label>
            <Input
              value={server.command || ""}
              onChange={(e) => set("command", e.target.value)}
              placeholder={t("mcpServers.commandPlaceholder")}
            />
          </div>
          <div>
            <label className="text-sm font-medium">{t("mcpServers.args")}</label>
            <Input
              value={server.args || ""}
              onChange={(e) => set("args", e.target.value)}
              placeholder={t("mcpServers.argsPlaceholder")}
            />
            <p className="text-[11px] text-muted-foreground mt-1">JSON array, e.g. ["--port", "8080"]</p>
          </div>
          <div>
            <label className="text-sm font-medium">{t("mcpServers.envs")}</label>
            <Input
              value={server.envs || ""}
              onChange={(e) => set("envs", e.target.value)}
              placeholder='{"KEY": "value"}'
            />
            <p className="text-[11px] text-muted-foreground mt-1">JSON object, e.g. {"{"}"NODE_ENV": "production"{"}"}</p>
          </div>
        </>
      ) : (
        <>
          <div>
            <label className="text-sm font-medium">{t("mcpServers.urlLabel")}</label>
            <Input
              value={server.url || ""}
              onChange={(e) => set("url", e.target.value)}
              placeholder={t("mcpServers.urlPlaceholder")}
            />
          </div>
          <div>
            <label className="text-sm font-medium">{t("mcpServers.authType")}</label>
            <select
              className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={server.auth ? JSON.parse(server.auth).type : "none"}
              onChange={(e) => {
                const authType = e.target.value;
                if (authType === "none") {
                  set("auth", undefined);
                } else {
                  const existing = server.auth ? JSON.parse(server.auth) : {};
                  set("auth", JSON.stringify({ type: authType, token: existing.token || "" }));
                }
              }}
            >
              <option value="none">{t("mcpServers.authNone")}</option>
              <option value="api_key">{t("mcpServers.authApiKey")}</option>
              <option value="token">{t("mcpServers.authToken")}</option>
            </select>
          </div>
          {(server.auth && JSON.parse(server.auth).type !== "none") && (
            <div>
              <label className="text-sm font-medium">{t("mcpServers.authValue")}</label>
              <Input
                value={JSON.parse(server.auth!).token || ""}
                onChange={(e) => {
                  const auth = JSON.parse(server.auth!);
                  auth.token = e.target.value;
                  set("auth", JSON.stringify(auth));
                }}
                placeholder={t("mcpServers.authValuePlaceholder")}
              />
            </div>
          )}
          <div>
            <label className="text-sm font-medium">{t("mcpServers.headers")}</label>
            <Input
              value={server.headers || ""}
              onChange={(e) => set("headers", e.target.value)}
              placeholder='{"Authorization": "Bearer ..."}'
            />
            <p className="text-[11px] text-muted-foreground mt-1">JSON object for additional HTTP headers</p>
          </div>
        </>
      )}
    </div>
  );
}

export default function MCPServersPage() {
  const { t } = useTranslation();
  const { data: servers, isLoading } = useMCPServers();
  const createServer = useCreateMCPServer();
  const deleteServer = useDeleteMCPServer();

  const [showCreate, setShowCreate] = useState(false);
  const [editTarget, setEditTarget] = useState<MCPServer | null>(null);
  const [form, setForm] = useState<Partial<MCPServer>>({ transport: "stdio", name: "" });
  const [deleteTarget, setDeleteTarget] = useState<MCPServer | null>(null);

  const updateServer = useUpdateMCPServer(editTarget?.id || "");

  const resetForm = () => setForm({ transport: "stdio", name: "" });

  const handleCreate = async () => {
    if (!form.name?.trim()) return;
    await createServer.mutateAsync({
      name: form.name.trim(),
      transport: form.transport || "stdio",
      url: form.url || undefined,
      auth: form.auth || undefined,
      headers: form.headers || undefined,
      command: form.command || undefined,
      args: form.args || undefined,
      envs: form.envs || undefined,
    });
    setShowCreate(false);
    resetForm();
  };

  const handleUpdate = async () => {
    if (!editTarget || !form.name?.trim()) return;
    await updateServer.mutateAsync({
      name: form.name.trim(),
      transport: form.transport,
      url: form.url || undefined,
      auth: form.auth || undefined,
      headers: form.headers || undefined,
      command: form.command || undefined,
      args: form.args || undefined,
      envs: form.envs || undefined,
    });
    setEditTarget(null);
    resetForm();
  };

  const openEdit = (s: MCPServer) => {
    setEditTarget(s);
    setForm({ ...s });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">{t("mcpServers.title")}</h1>
        <Button onClick={() => { resetForm(); setShowCreate(true); }}>
          <Plus className="w-4 h-4 mr-1" />
          {t("mcpServers.newServer")}
        </Button>
      </div>

      {isLoading && <p className="text-sm text-muted-foreground">{t("common.loading")}</p>}

      {!isLoading && servers && servers.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Server className="w-12 h-12 text-muted-foreground/50 mb-4" />
          <h2 className="text-lg font-medium">{t("mcpServers.noServers")}</h2>
          <p className="text-sm text-muted-foreground mt-1 max-w-sm">{t("mcpServers.noServersDesc")}</p>
          <Button className="mt-4" onClick={() => { resetForm(); setShowCreate(true); }}>
            <Plus className="w-4 h-4 mr-1" />
            {t("mcpServers.createTitle")}
          </Button>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {(servers || []).map((server) => (
          <Card key={server.id} className="p-4 hover:shadow-md transition-shadow cursor-pointer group" onClick={() => openEdit(server)}>
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <h3 className="font-medium truncate">{server.name}</h3>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-muted text-muted-foreground uppercase">
                    {TRANSPORT_LABELS[server.transport] || server.transport}
                  </span>
                  <span className="text-sm text-muted-foreground truncate">
                    {server.transport === "stdio"
                      ? (server.command || "—")
                      : (server.url || "—")}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between mt-3 pt-3 border-t">
              <span className="text-xs text-muted-foreground">{timeAgo(server.updated_at)}</span>
              <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  className="p-1 hover:bg-muted rounded"
                  onClick={(e) => { e.stopPropagation(); openEdit(server); }}
                >
                  <Pencil className="w-3.5 h-3.5" />
                </button>
                <button
                  className="p-1 hover:bg-destructive/10 rounded text-destructive"
                  onClick={(e) => { e.stopPropagation(); setDeleteTarget(server); }}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Create Dialog */}
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("mcpServers.createTitle")}</DialogTitle>
          </DialogHeader>
          <ServerForm server={form} onChange={setForm} t={t} />
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setShowCreate(false)}>{t("common.cancel")}</Button>
            <Button onClick={handleCreate} disabled={!form.name?.trim() || createServer.isPending}>
              {createServer.isPending ? t("mcpServers.saving") : t("common.save")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editTarget} onOpenChange={() => setEditTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("mcpServers.editTitle")}</DialogTitle>
          </DialogHeader>
          <ServerForm server={form} onChange={setForm} t={t} />
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setEditTarget(null)}>{t("common.cancel")}</Button>
            <Button onClick={handleUpdate} disabled={!form.name?.trim() || updateServer.isPending}>
              {updateServer.isPending ? t("mcpServers.saving") : t("common.save")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={!!deleteTarget}
        onCancel={() => setDeleteTarget(null)}
        title={t("mcpServers.deleteTitle")}
        description={t("mcpServers.deleteDesc", { name: deleteTarget?.name || "" })}
        onConfirm={() => {
          if (deleteTarget) {
            deleteServer.mutate(deleteTarget.id);
            setDeleteTarget(null);
          }
        }}
      />
    </div>
  );
}
