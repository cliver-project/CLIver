import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/status-pill";
import { useStatus, useAdapters } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

function fmtUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export default function DashboardPage() {
  const { t } = useTranslation();
  const { data: status } = useStatus();
  const { data: adapters } = useAdapters();

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{t("dashboard.title")}</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">{t("dashboard.uptime")}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {status ? fmtUptime(Number(status.uptime ?? 0)) : "—"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">{t("dashboard.tasksRun")}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{String(status?.tasks_run ?? "—")}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">{t("dashboard.status")}</CardTitle>
          </CardHeader>
          <CardContent>
            <StatusPill status={status ? "active" : "inactive"} />
          </CardContent>
        </Card>
      </div>
      {adapters && (
        <div>
          <h2 className="text-lg font-semibold mb-3">{t("dashboard.platformAdapters")}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(adapters as Array<Record<string, unknown>>).map(
              (a, i) => (
                <Card key={i}>
                  <CardContent className="pt-4">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{String(a.name ?? a.type)}</span>
                      <StatusPill status={String(a.status ?? "unknown")} />
                    </div>
                  </CardContent>
                </Card>
              ),
            )}
          </div>
        </div>
      )}
    </div>
  );
}
