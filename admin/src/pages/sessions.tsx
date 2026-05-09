import { useState } from "react";
import { useSearchParams } from "react-router";
import { Link } from "react-router";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useSessions, useDeleteSession } from "@/hooks/use-api";
import { MessageSquare, ChevronRight, Trash2 } from "lucide-react";
import { useTranslation } from "@/i18n";

function SessionList({ source }: { source: "cli" | "gateway" }) {
  const { t } = useTranslation();
  const { data, isLoading } = useSessions(source);
  const deleteSession = useDeleteSession(source);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  if (isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;

  const sessions = (data ?? []) as Array<Record<string, unknown>>;
  if (sessions.length === 0) {
    return <p className="text-muted-foreground">{t("sessions.noSessions")}</p>;
  }

  return (
    <>
      <ConfirmDialog
        open={deleteTarget !== null}
        title={t("sessions.deleteSession")}
        description={t("sessions.deleteSessionDescription")}
        destructive
        onConfirm={() => {
          if (deleteTarget) deleteSession.mutate(deleteTarget);
          setDeleteTarget(null);
        }}
        onCancel={() => setDeleteTarget(null)}
      />
      <div className="space-y-2">
        {sessions.map((s, i) => {
          const id = String(s.id ?? s.session_id ?? i);
          const title =
            String(s.display_title ?? s.title ?? "").trim() || t("sessions.untitled");
          const turnCount = Number(s.turn_count ?? 0);
          const updatedAt = String(s.updated_at ?? s.created_at ?? "");

          return (
            <Card key={id} className={`hover:bg-accent/50 transition-colors ${i % 2 === 1 ? "bg-muted/30" : ""}`}>
              <CardContent className="py-3">
                <div className="flex items-center gap-3">
                  <Link
                    to={`/admin/sessions/${source}/${encodeURIComponent(id)}?title=${encodeURIComponent(title)}`}
                    className="flex items-center gap-3 flex-1 min-w-0"
                  >
                    <div className="flex items-center justify-center w-8 h-8 rounded-md bg-muted text-muted-foreground shrink-0">
                      <MessageSquare className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{title}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-xs text-muted-foreground font-mono truncate max-w-[200px]">
                          {id}
                        </span>
                        {turnCount > 0 && (
                          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                            {t("sessions.turns", { count: turnCount })}
                          </Badge>
                        )}
                      </div>
                    </div>
                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                      {updatedAt}
                    </span>
                    <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
                  </Link>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={() => setDeleteTarget(id)}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </>
  );
}

export default function SessionsPage() {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();
  const tab = searchParams.get("tab") || "cli";

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{t("sessions.title")}</h1>
      <Tabs
        value={tab}
        onValueChange={(v) => setSearchParams({ tab: v })}
      >
        <TabsList>
          <TabsTrigger value="cli">{t("sessions.cli")}</TabsTrigger>
          <TabsTrigger value="gateway">{t("sessions.gateway")}</TabsTrigger>
        </TabsList>
        <TabsContent value="cli">
          <SessionList source="cli" />
        </TabsContent>
        <TabsContent value="gateway">
          <SessionList source="gateway" />
        </TabsContent>
      </Tabs>
    </div>
  );
}
