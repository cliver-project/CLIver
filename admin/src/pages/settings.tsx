import { PageLayout } from "@/components/layout/PageLayout";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useConfig, useAdapters, useScenarios, useInstallScenario, useRemoveScenario } from "@/hooks/use-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/status-pill";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plus, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { useTranslation } from "@/i18n";

export default function SettingsPage() {
  const { t } = useTranslation();
  const { data: config } = useConfig();
  const { data: adapters } = useAdapters();
  const { data: scenarios } = useScenarios();
  const installScenario = useInstallScenario();
  const removeScenario = useRemoveScenario();

  const [showInstall, setShowInstall] = useState(false);
  const [installSource, setInstallSource] = useState("");
  const [removeTarget, setRemoveTarget] = useState<string | null>(null);

  const handleInstall = async () => {
    if (!installSource.trim()) return;
    try {
      await installScenario.mutateAsync(installSource.trim());
      setShowInstall(false);
      setInstallSource("");
    } catch (err) {
      console.error("Install failed:", err);
    }
  };

  const handleRemove = async () => {
    if (!removeTarget) return;
    try {
      await removeScenario.mutateAsync(removeTarget);
      setRemoveTarget(null);
    } catch (err) {
      console.error("Remove failed:", err);
    }
  };

  return (
    <PageLayout title={t("settings.title")}>
      <Tabs defaultValue="config">
        <TabsList>
          <TabsTrigger value="config">{t("settings.configuration")}</TabsTrigger>
          <TabsTrigger value="agents">{t("settings.agents")}</TabsTrigger>
          <TabsTrigger value="adapters">{t("settings.adapters")}</TabsTrigger>
          <TabsTrigger value="scenarios">{t("settings.scenarios")}</TabsTrigger>
        </TabsList>

        <TabsContent value="config" className="mt-4">
          {config && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">{t("settings.configuration")}</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-xs bg-muted p-3 rounded-md overflow-auto max-h-96">
                  {JSON.stringify(config, null, 2)}
                </pre>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="agents" className="mt-4">
          <div className="space-y-4">
            {config?.agents && typeof config.agents === 'object' ? (
              Object.entries(config.agents as Record<string, Record<string, unknown>>).map(
                ([name, agent]) => (
                  <Card key={name}>
                    <CardHeader>
                      <CardTitle className="text-sm">{name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="text-muted-foreground">{t("settings.type")}</div>
                        <div>{String(agent.type || "cliver")}</div>
                        <div className="text-muted-foreground">{t("settings.model")}</div>
                        <div>{String(agent.model || "default")}</div>
                        {agent.role != null && (
                          <>
                            <div className="text-muted-foreground">{t("settings.role")}</div>
                            <div>{String(agent.role)}</div>
                          </>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ),
              )
            ) : (
              <p className="text-sm text-muted-foreground">{t("settings.noAgents")}</p>
            )}
          </div>
        </TabsContent>

        <TabsContent value="adapters" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {adapters && Array.isArray(adapters) ? (
              adapters.map((adapter: Record<string, unknown>) => {
                const name = String(adapter.name || "");
                const type = String(adapter.type || "");
                const status = adapter.status ? String(adapter.status) : "inactive";
                return (
                  <Card key={name}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm">{name}</CardTitle>
                        <StatusPill status={status} />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-xs text-muted-foreground">{t("settings.type")}: {type}</p>
                    </CardContent>
                  </Card>
                );
              })
            ) : (
              <p className="text-sm text-muted-foreground">{t("settings.noAdapters")}</p>
            )}
          </div>
        </TabsContent>

        <TabsContent value="scenarios" className="mt-4">
          <div className="flex items-center justify-between mb-4">
            <div className="text-sm text-muted-foreground">
              {scenarios ? t("settings.scenariosInstalled", { count: (scenarios as unknown[]).length }) : t("common.loading")}
            </div>
            <Button size="sm" onClick={() => setShowInstall(true)}>
              <Plus className="w-4 h-4 mr-1.5" />
              {t("settings.installFromGithub")}
            </Button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {scenarios && Array.isArray(scenarios) ? (
              scenarios.map((s: Record<string, unknown>) => (
                <Card key={String(s.id)}>
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm">{String(s.display_name)}</CardTitle>
                      <div className="flex items-center gap-1.5">
                        <span className={cn(
                          "text-[10px] px-1.5 py-0.5 rounded font-medium",
                          s.source === "builtin" ? "bg-primary/10 text-primary" : "bg-emerald-50 text-emerald-700"
                        )}>
                          {String(s.source)}
                        </span>
                        {s.source !== "builtin" && (
                          <button
                            onClick={() => setRemoveTarget(String(s.id))}
                            className="p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-xs text-muted-foreground mb-2">{String(s.description || "")}</p>
                    {Array.isArray(s.tags) && s.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {(s.tags as string[]).map((tag) => (
                          <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))
            ) : (
              <p className="text-sm text-muted-foreground">{t("settings.noScenarios")}</p>
            )}
          </div>

          <Dialog open={showInstall} onOpenChange={setShowInstall}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>{t("settings.installTitle")}</DialogTitle>
                <DialogDescription>{t("settings.installDesc")}</DialogDescription>
              </DialogHeader>
              <div className="py-2">
                <Label htmlFor="scenario-source">{t("settings.githubSource")}</Label>
                <Input
                  id="scenario-source"
                  placeholder={t("settings.githubPlaceholder")}
                  value={installSource}
                  onChange={(e) => setInstallSource(e.target.value)}
                />
                <p className="text-xs text-muted-foreground mt-1">{t("settings.githubExample")}</p>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setShowInstall(false)}>{t("common.cancel")}</Button>
                <Button onClick={handleInstall} disabled={!installSource.trim() || installScenario.isPending}>
                  {installScenario.isPending ? t("settings.installing") : t("settings.install")}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <ConfirmDialog
            open={!!removeTarget}
            title={t("settings.removeTitle")}
            description={t("settings.removeDesc", { name: removeTarget || "" })}
            onConfirm={handleRemove}
            onCancel={() => setRemoveTarget(null)}
            destructive={true}
          />
        </TabsContent>
      </Tabs>
    </PageLayout>
  );
}
