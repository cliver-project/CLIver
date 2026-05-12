import { PageLayout } from "@/components/layout/PageLayout";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useConfig, useAdapters } from "@/hooks/use-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/status-pill";

export default function SettingsPage() {
  const { data: config } = useConfig();
  const { data: adapters } = useAdapters();

  return (
    <PageLayout title="Settings">
      <Tabs defaultValue="config">
        <TabsList>
          <TabsTrigger value="config">Configuration</TabsTrigger>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="adapters">Adapters</TabsTrigger>
        </TabsList>

        <TabsContent value="config" className="mt-4">
          {config && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Configuration</CardTitle>
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
                        <div className="text-muted-foreground">Type</div>
                        <div>{String(agent.type || "cliver")}</div>
                        <div className="text-muted-foreground">Model</div>
                        <div>{String(agent.model || "default")}</div>
                        {agent.role != null && (
                          <>
                            <div className="text-muted-foreground">Role</div>
                            <div>{String(agent.role)}</div>
                          </>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ),
              )
            ) : (
              <p className="text-sm text-muted-foreground">No agents configured.</p>
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
                      <p className="text-xs text-muted-foreground">Type: {type}</p>
                    </CardContent>
                  </Card>
                );
              })
            ) : (
              <p className="text-sm text-muted-foreground">No adapters configured.</p>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </PageLayout>
  );
}
