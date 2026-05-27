import { useState } from "react";
import { PageLayout } from "@/components/layout/PageLayout";
import { useAdapters, useConfig, useSaveConfig } from "@/hooks/use-api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import { StatusPill } from "@/components/status-pill";
import { Plus, Trash2 } from "lucide-react";
import { useTranslation } from "@/i18n";

const ADAPTER_TYPES = ["slack", "telegram", "discord", "feishu", "wechat"];

interface Adapter {
  name: string;
  type: string;
  state?: string;
  token?: string;
  app_token?: string;
  home_channel?: string;
  allowed_users?: string[];
  error?: string;
}

export default function AdaptersList() {
  const { t } = useTranslation();
  const { data: rawAdapters, isLoading } = useAdapters();
  const { data: config } = useConfig();
  const saveConfig = useSaveConfig();
  const [showAdd, setShowAdd] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [form, setForm] = useState({
    name: "",
    type: "slack",
    token: "",
    app_token: "",
    home_channel: "",
  });

  const adapters: Adapter[] = Array.isArray(rawAdapters)
    ? (rawAdapters as unknown as Adapter[])
    : [];

  const savePlatforms = (
    newPlatforms: Record<string, Record<string, unknown>>,
  ) => {
    if (!config) return;
    const updated = { ...(config as Record<string, unknown>) };
    const gw = { ...((updated.gateway || {}) as Record<string, unknown>) };
    gw.platforms = newPlatforms;
    updated.gateway = gw;
    saveConfig.mutate(updated);
  };

  const getPlatforms = (): Record<string, Record<string, unknown>> => {
    const cfg = config as Record<string, unknown> | undefined;
    const gw = cfg?.gateway as Record<string, unknown> | undefined;
    return (gw?.platforms as Record<string, Record<string, unknown>>) || {};
  };

  const handleAdd = () => {
    if (!form.name.trim() || !form.type) return;
    const platforms = getPlatforms();
    const newPlatform: Record<string, unknown> = { type: form.type, token: form.token };
    if (form.app_token) newPlatform.app_token = form.app_token;
    if (form.home_channel) newPlatform.home_channel = form.home_channel;
    savePlatforms({ ...platforms, [form.name]: newPlatform });
    setShowAdd(false);
    setForm({ name: "", type: "slack", token: "", app_token: "", home_channel: "" });
  };

  const handleDelete = () => {
    if (!deleteTarget) return;
    const platforms = getPlatforms();
    const copy = { ...platforms };
    delete copy[deleteTarget];
    savePlatforms(copy);
    setDeleteTarget(null);
  };

  return (
    <PageLayout
      title={t("sidebar.adapters")}
      actions={
        <Button
          size="sm"
          onClick={() => {
            setForm({ name: "", type: "slack", token: "", app_token: "", home_channel: "" });
            setShowAdd(true);
          }}
        >
          <Plus className="w-4 h-4 mr-1.5" />
          {t("common.add")}
        </Button>
      }
    >
      {isLoading ? (
        <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
      ) : adapters.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-sm text-muted-foreground">
            No adapters configured. Click &quot;Add&quot; to connect a messaging platform.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {adapters.map((a) => (
            <Card key={a.name} className="group">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">{a.name}</CardTitle>
                  <div className="flex items-center gap-1.5">
                    <StatusPill status={a.state === "connected" ? "active" : "inactive"} />
                    <button
                      onClick={() => setDeleteTarget(a.name)}
                      className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-all"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground space-y-1">
                  <div>
                    Type: <span className="text-foreground">{a.type}</span>
                  </div>
                  <div>
                    Token: <span className="text-foreground">{(a.token || "").substring(0, 8)}****</span>
                  </div>
                  {a.home_channel ? (
                    <div>
                      Channel: <span className="text-foreground">{a.home_channel}</span>
                    </div>
                  ) : null}
                  {a.error ? (
                    <div className="text-red-600">{a.error}</div>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Add Dialog */}
      <Dialog open={showAdd} onOpenChange={setShowAdd}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Adapter</DialogTitle>
            <DialogDescription>Connect a messaging platform to CLIver.</DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div>
              <Label>Name</Label>
              <Input
                placeholder="e.g. my-slack"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>
            <div>
              <Label>Type</Label>
              <Select value={form.type} onValueChange={(v) => setForm({ ...form, type: v })}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ADAPTER_TYPES.map((tp) => (
                    <SelectItem key={tp} value={tp}>
                      {tp}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Token</Label>
              <Input
                type="password"
                placeholder="Bot token"
                value={form.token}
                onChange={(e) => setForm({ ...form, token: e.target.value })}
              />
            </div>
            <div>
              <Label>App Token (optional)</Label>
              <Input
                type="password"
                placeholder="App-level token (Slack only)"
                value={form.app_token}
                onChange={(e) => setForm({ ...form, app_token: e.target.value })}
              />
            </div>
            <div>
              <Label>Home Channel (optional)</Label>
              <Input
                placeholder="Channel ID"
                value={form.home_channel}
                onChange={(e) => setForm({ ...form, home_channel: e.target.value })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAdd(false)}>
              {t("common.cancel")}
            </Button>
            <Button onClick={handleAdd} disabled={!form.name.trim() || !form.token.trim()}>
              {t("common.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Adapter"
        description={`Remove adapter "${deleteTarget}"? Restart the gateway for changes to take effect.`}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
        destructive
      />
    </PageLayout>
  );
}
