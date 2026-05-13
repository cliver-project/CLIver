import { PageLayout } from "@/components/layout/PageLayout";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useConfig, useAdapters } from "@/hooks/use-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/status-pill";
import { useTranslation } from "@/i18n";

export default function SettingsPage() {
  const { t } = useTranslation();
  const { data: config } = useConfig();
  const { data: adapters } = useAdapters();

  return (
    <PageLayout title={t("settings.title")}>
      <Tabs defaultValue="config">
        <TabsList>
          <TabsTrigger value="config">{t("settings.configuration")}</TabsTrigger>
          <TabsTrigger value="adapters">{t("settings.adapters")}</TabsTrigger>
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
      </Tabs>
    </PageLayout>
  );
}
