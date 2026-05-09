import { useState } from "react";
import { Link } from "react-router";
import { Workflow, Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StatusPill } from "@/components/status-pill";
import { useWorkflows, useExecutions } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

export default function WorkflowListPage() {
  const { t } = useTranslation();
  const [tab, setTab] = useState("definitions");

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{t("workflows.title")}</h1>
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="definitions">{t("workflows.definitions")}</TabsTrigger>
          <TabsTrigger value="executions">{t("workflows.executions")}</TabsTrigger>
        </TabsList>
        <TabsContent value="definitions">
          <WorkflowGrid />
        </TabsContent>
        <TabsContent value="executions">
          <ExecutionList />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function WorkflowGrid() {
  const { t } = useTranslation();
  const { data, isLoading } = useWorkflows();
  if (isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;

  const workflows = (data ?? []) as Array<{
    name: string;
    description?: string;
    steps: number;
    source?: string;
  }>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {workflows.map((wf) => (
        <Link key={wf.name} to={`/admin/workflows/${encodeURIComponent(wf.name)}`}>
          <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Workflow className="w-4 h-4 text-primary" />
                {wf.name}
                {wf.source === "project" && (
                  <Badge variant="secondary" className="text-[10px]">project</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-2">
                {wf.description || t("common.noDescription")}
              </p>
              <p className="text-xs text-muted-foreground">{t("common.steps", { count: wf.steps })}</p>
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  );
}

function ExecutionList() {
  const { t } = useTranslation();
  const { data, isLoading } = useExecutions();
  if (isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;

  const executions = (data ?? []) as Array<Record<string, unknown>>;
  if (executions.length === 0) {
    return <p className="text-muted-foreground">{t("workflows.noExecutions")}</p>;
  }

  return (
    <div className="space-y-2">
      {executions.map((e, i) => (
        <Card key={i} className={i % 2 === 1 ? "bg-muted/30" : ""}>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="font-medium">{String(e.workflow_name)}</span>
                <span className="text-xs text-muted-foreground ml-2 font-mono">
                  {String(e.thread_id ?? "")}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {String(e.started_at ?? "")}
                </span>
                <StatusPill status={String(e.status ?? "unknown")} />
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
