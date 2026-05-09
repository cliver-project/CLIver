import { useState } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useSessionTurns, useDeleteSession } from "@/hooks/use-api";
import { ConfirmDialog } from "@/components/confirm-dialog";
import {
  ArrowLeft, Trash2, User, Bot, Wrench, Brain, ChevronDown, ChevronRight, Image as ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MarkdownView } from "@/components/markdown-view";
import { useTranslation } from "@/i18n";

interface TurnData {
  role: string;
  content: string;
  timestamp?: string;
  type?: string;
  additional_kwargs?: Record<string, unknown>;
  tool_calls?: Array<{ name: string; args: Record<string, unknown>; id?: string }>;
  tool_call_id?: string;
  tool_name?: string;
  media?: Array<{ type: string; path: string; mime?: string }>;
}

function ReasoningBlock({ content }: { content: string }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  if (!content) return null;
  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <Brain className="w-3 h-3" />
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {t("sessions.thinking")}
      </button>
      {open && (
        <div className="mt-1 pl-4 border-l-2 border-muted text-xs text-muted-foreground italic whitespace-pre-wrap max-h-60 overflow-auto">
          {content}
        </div>
      )}
    </div>
  );
}

function ToolCallsBlock({ calls }: { calls: TurnData["tool_calls"] }) {
  if (!calls || calls.length === 0) return null;
  return (
    <div className="mt-2 space-y-1">
      {calls.map((tc, i) => (
        <div key={i} className="flex items-center gap-2 text-xs">
          <Wrench className="w-3 h-3 text-muted-foreground" />
          <Badge variant="outline" className="text-[10px]">{tc.name}</Badge>
          {tc.args && Object.keys(tc.args).length > 0 && (
            <span className="text-muted-foreground font-mono truncate max-w-[400px]">
              {JSON.stringify(tc.args).slice(0, 100)}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

function MediaBlock({ media, sessionSource, sessionId }: {
  media: TurnData["media"]; sessionSource?: string; sessionId?: string;
}) {
  if (!media || media.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {media.map((m, i) => {
        const isImage = m.type === "image" || m.mime?.startsWith("image/");
        if (isImage && m.path) {
          const src = m.path.startsWith("http") ? m.path : `/admin/api/sessions/${sessionSource}/${sessionId}/media/${m.path}`;
          return <img key={i} src={src} alt="" className="max-w-xs max-h-48 rounded-md border" />;
        }
        return (
          <div key={i} className="flex items-center gap-1 text-xs text-muted-foreground">
            <ImageIcon className="w-3 h-3" />
            <span>{m.path || m.type}</span>
          </div>
        );
      })}
    </div>
  );
}

function TurnCard({ turn, source, sessionId }: { turn: TurnData; source?: string; sessionId?: string }) {
  const isUser = turn.role === "user";
  const isTool = turn.role === "tool";
  const reasoning = turn.additional_kwargs?.reasoning_content as string | undefined;

  if (isTool) {
    return (
      <Card className="border-l-4 border-l-amber-500/50">
        <CardContent className="pt-3 pb-3">
          <div className="flex items-start gap-3">
            <div className="flex items-center justify-center w-7 h-7 rounded-full shrink-0 mt-0.5 bg-amber-500/10 text-amber-600">
              <Wrench className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium">Tool Result</span>
                {turn.tool_name && <Badge variant="outline" className="text-[10px]">{turn.tool_name}</Badge>}
                {turn.timestamp && <span className="text-xs text-muted-foreground">{turn.timestamp}</span>}
              </div>
              <pre className="text-xs text-muted-foreground whitespace-pre-wrap max-h-40 overflow-auto bg-muted/30 rounded p-2">
                {turn.content}
              </pre>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn(isUser ? "border-l-4 border-l-primary" : "")}>
      <CardContent className="pt-4">
        <div className="flex items-start gap-3">
          <div className={cn(
            "flex items-center justify-center w-7 h-7 rounded-full shrink-0 mt-0.5",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
          )}>
            {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium capitalize">{turn.role}</span>
              {turn.timestamp && <span className="text-xs text-muted-foreground">{turn.timestamp}</span>}
            </div>
            {reasoning && <ReasoningBlock content={reasoning} />}
            <MarkdownView content={turn.content} />
            <ToolCallsBlock calls={turn.tool_calls} />
            <MediaBlock media={turn.media} sessionSource={source} sessionId={sessionId} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function SessionDetailPage() {
  const { t } = useTranslation();
  const { source, id } = useParams<{ source: string; id: string }>();
  const [searchParams] = useSearchParams();
  const title = searchParams.get("title");
  const navigate = useNavigate();
  const { data: turns, isLoading } = useSessionTurns(source!, id!);
  const deleteSession = useDeleteSession(source!);
  const [confirmOpen, setConfirmOpen] = useState(false);

  async function handleDelete() {
    await deleteSession.mutateAsync(id!);
    navigate(`/admin/sessions?tab=${source}`);
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={() => navigate(`/admin/sessions?tab=${source}`)}>
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold truncate">{title || t("sessions.session")}</h1>
          <p className="text-xs text-muted-foreground font-mono truncate">{source} / {id}</p>
        </div>
        <Button variant="destructive" size="sm" onClick={() => setConfirmOpen(true)} disabled={deleteSession.isPending}>
          <Trash2 className="w-4 h-4 mr-1" />
          {t("common.delete")}
        </Button>
      </div>

      <ConfirmDialog
        open={confirmOpen}
        title={t("sessions.deleteSession")}
        description={t("sessions.deleteSessionDescription")}
        destructive
        onConfirm={() => { setConfirmOpen(false); handleDelete(); }}
        onCancel={() => setConfirmOpen(false)}
      />

      {isLoading && <p className="text-muted-foreground">{t("common.loading")}</p>}
      {turns && turns.length === 0 && <p className="text-muted-foreground">{t("sessions.noConversationTurns")}</p>}

      {turns && turns.length > 0 && (
        <div className="space-y-3">
          {(turns as TurnData[]).map((turn, i) => (
            <TurnCard key={i} turn={turn} source={source} sessionId={id} />
          ))}
        </div>
      )}
    </div>
  );
}
