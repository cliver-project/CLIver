import { useState } from "react";
import { useNavigate } from "react-router";
import { PageLayout } from "@/components/layout/PageLayout";
import { useScenarios, useInstallScenario, useRemoveScenario } from "@/hooks/use-api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription,
} from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Trash2 } from "lucide-react";
import { useTranslation } from "@/i18n";
import { cn } from "@/lib/utils";

export default function ScenariosList() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { data: scenarios, isLoading } = useScenarios();
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
    <PageLayout
      title={t("scenarios.title")}
      actions={
        <Button size="sm" onClick={() => setShowInstall(true)}>
          <Plus className="w-4 h-4 mr-1.5" />
          {t("scenarios.installFromGithub")}
        </Button>
      }
    >
      {isLoading ? (
        <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
      ) : !scenarios || (scenarios as unknown[]).length === 0 ? (
        <div className="text-center py-12">
          <p className="text-sm text-muted-foreground">{t("scenarios.noScenarios")}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {(scenarios as Array<Record<string, unknown>>).map((s) => (
            <Card
              key={String(s.id)}
              className="cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => navigate(`/admin/scenarios/${String(s.id)}`)}
            >
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
                        onClick={(e) => { e.stopPropagation(); setRemoveTarget(String(s.id)); }}
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
          ))}
        </div>
      )}

      {/* Install Dialog */}
      <Dialog open={showInstall} onOpenChange={setShowInstall}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("scenarios.installTitle")}</DialogTitle>
            <DialogDescription>{t("scenarios.installDesc")}</DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <Label htmlFor="scenario-source">{t("scenarios.githubSource")}</Label>
            <Input
              id="scenario-source"
              placeholder={t("scenarios.githubPlaceholder")}
              value={installSource}
              onChange={(e) => setInstallSource(e.target.value)}
            />
            <p className="text-xs text-muted-foreground mt-1">{t("scenarios.githubExample")}</p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowInstall(false)}>{t("common.cancel")}</Button>
            <Button onClick={handleInstall} disabled={!installSource.trim() || installScenario.isPending}>
              {installScenario.isPending ? t("scenarios.installing") : t("scenarios.install")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Remove Confirmation */}
      <ConfirmDialog
        open={!!removeTarget}
        title={t("scenarios.removeTitle")}
        description={t("scenarios.removeDesc", { name: removeTarget || "" })}
        onConfirm={handleRemove}
        onCancel={() => setRemoveTarget(null)}
        destructive={true}
      />
    </PageLayout>
  );
}
