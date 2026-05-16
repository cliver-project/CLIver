import { useParams, useNavigate } from "react-router";
import { useState } from "react";
import { PageLayout } from "@/components/layout/PageLayout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription,
} from "@/components/ui/dialog";
import { useTranslation } from "@/i18n";
import { useQuery } from "@tanstack/react-query";
import { api, apiPost } from "@/lib/api";
import { Book } from "lucide-react";

interface ScenarioDetail {
  id: string;
  name: string;
  display_name: string;
  description: string;
  tags: string[];
  agent_requirements: string[];
  source: string;
  template?: {
    "$schema": string;
    title: string;
    description: string;
    default_agent: string;
    cells: Array<{ id: string; type: string; title: string }>;
  };
}

export default function ScenarioDetailPage() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [showCreate, setShowCreate] = useState(false);
  const [labTitle, setLabTitle] = useState("");

  const { data: scenario, isLoading } = useQuery({
    queryKey: ["scenario", id],
    queryFn: () => api<ScenarioDetail>(`/scenarios/${encodeURIComponent(id || "")}`),
    enabled: !!id,
  });

  const handleCreateLab = async () => {
    if (!scenario || !labTitle.trim()) return;
    try {
      const nb = await apiPost<{ id: string }>("/labs", {
        title: labTitle,
        description: scenario.description,
        scenario_id: scenario.id,
        default_agent: scenario.template?.default_agent,
        cells: scenario.template?.cells || [],
      });
      navigate(`/admin/labs/${nb.id}`);
    } catch (e) {
      console.error("Failed to create lab:", e);
    }
  };

  if (isLoading) {
    return <PageLayout title="..."><p className="text-sm text-muted-foreground">{t("common.loading")}</p></PageLayout>;
  }

  if (!scenario) {
    return <PageLayout title="Not Found"><p className="text-sm text-muted-foreground">Scenario not found.</p></PageLayout>;
  }

  const cells = scenario.template?.cells || [];

  return (
    <PageLayout
      title={scenario.display_name}
      breadcrumb={[
        { label: t("scenarios.title"), href: "/admin/scenarios" },
        { label: scenario.display_name },
      ]}
      actions={
        <Button onClick={() => { setLabTitle(scenario.display_name); setShowCreate(true); }}>
          <Book className="w-4 h-4 mr-1.5" />
          Create AI Lab
        </Button>
      }
    >
      <div className="space-y-6 max-w-3xl">
        {/* Info */}
        <Card>
          <CardContent className="pt-6">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Source</span>
                <p className="font-medium">{scenario.source}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Required Agents</span>
                <p className="font-medium">{scenario.agent_requirements.join(", ") || "Any"}</p>
              </div>
            </div>
            {scenario.description && (
              <p className="mt-4 text-sm text-muted-foreground">{scenario.description}</p>
            )}
            {scenario.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-3">
                {scenario.tags.map((tag) => (
                  <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{tag}</span>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Template cells preview */}
        {cells.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold mb-3">Lab Template ({cells.length} cells)</h3>
            <div className="space-y-2">
              {cells.map((cell, idx) => {
                const colors: Record<string, string> = {
                  config: "border-l-indigo-500",
                  llm: "border-l-purple-500",
                  code: "border-l-emerald-500",
                  display: "border-l-amber-500",
                };
                const icons: Record<string, string> = {
                  config: "⚙", llm: "🤖", code: "</>;", display: "📄",
                };
                return (
                  <div key={cell.id} className={`bg-card border rounded-md border-l-[3px] ${colors[cell.type] || ""} px-3 py-2 flex items-center gap-3`}>
                    <span className="text-xs text-muted-foreground w-5">{idx + 1}</span>
                    <span className="text-sm">{icons[cell.type] || "?"}</span>
                    <span className="text-sm font-medium flex-1">{cell.title}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{cell.type}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Create AI Lab Dialog */}
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create AI Lab from Scenario</DialogTitle>
            <DialogDescription>
              Create a new lab using the "{scenario.display_name}" template.
            </DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <Label>Lab Title</Label>
            <Input
              value={labTitle}
              onChange={(e) => setLabTitle(e.target.value)}
              placeholder="e.g. My Research Project"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreate(false)}>{t("common.cancel")}</Button>
            <Button onClick={handleCreateLab} disabled={!labTitle.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </PageLayout>
  );
}
