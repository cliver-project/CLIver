import { PageLayout } from "@/components/layout/PageLayout";
import { useAdapters } from "@/hooks/use-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/status-pill";
import { useTranslation } from "@/i18n";

export default function AdaptersList() {
  const { t } = useTranslation();
  const { data: adapters, isLoading } = useAdapters();

  return (
    <PageLayout title={t("sidebar.adapters")}>
      {isLoading ? (
        <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
      ) : !adapters || (adapters as unknown[]).length === 0 ? (
        <div className="text-center py-12">
          <p className="text-sm text-muted-foreground">
            No adapters configured. Add platform adapters in Settings → Gateway.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {(adapters as Array<Record<string, unknown>>).map((a) => (
            <Card key={String(a.name)}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">{String(a.name)}</CardTitle>
                  <StatusPill status={String(a.status || "inactive")} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground space-y-1">
                  <div>Type: {String(a.type || "unknown")}</div>
                  {a.home_channel ? <div>Channel: {String(a.home_channel)}</div> : null}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </PageLayout>
  );
}
