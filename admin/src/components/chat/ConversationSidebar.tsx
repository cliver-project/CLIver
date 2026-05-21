import { memo, useState, useRef, useCallback } from "react";
import { Link } from "react-router";
import { Plus, Trash2, MessageSquare, Pencil, Check, X } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useTranslation } from "@/i18n";
import { useConversations, useDeleteConversation, useUpdateConversationTitle } from "@/hooks/use-conversations";
import { cn } from "@/lib/utils";

function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60_000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);
  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

interface ConversationSidebarProps {
  activeId: string | null;
  runningId?: string | null;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export const ConversationSidebar = memo(function ConversationSidebar({
  activeId,
  runningId,
  onNew,
  onDelete,
}: ConversationSidebarProps) {
  const { t } = useTranslation();
  const { data: conversations, isLoading } = useConversations();
  const deleteConv = useDeleteConversation();
  const updateTitle = useUpdateConversationTitle();
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const editRef = useRef<HTMLInputElement>(null);
  const list = conversations || [];

  const handleDeleteConfirm = useCallback(() => {
    if (!deleteConfirmId) return;
    deleteConv.mutate(deleteConfirmId);
    onDelete(deleteConfirmId);
    setDeleteConfirmId(null);
  }, [deleteConfirmId, deleteConv, onDelete]);

  const startRename = useCallback((id: string, currentTitle: string | null) => {
    setEditingId(id);
    setEditValue(currentTitle || "");
    setTimeout(() => editRef.current?.focus(), 0);
  }, []);

  const commitRename = useCallback(() => {
    if (!editingId) return;
    const trimmed = editValue.trim();
    if (trimmed) {
      updateTitle.mutate({ id: editingId, title: trimmed });
    }
    setEditingId(null);
  }, [editingId, editValue, updateTitle]);

  const cancelRename = useCallback(() => {
    setEditingId(null);
  }, []);

  return (
    <aside className="w-[260px] flex flex-col border-r border-sidebar-border bg-sidebar-background shrink-0">
      {/* New Chat button */}
      <div className="p-3 border-b border-sidebar-border">
        <button
          type="button"
          onClick={onNew}
          className="flex items-center justify-center gap-2 w-full rounded-lg border border-border px-3 py-2 text-sm font-medium hover:bg-secondary transition-colors cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          {t("chat.newChat")}
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="p-3 space-y-2">
          {isLoading && list.length === 0 && (
            <div className="px-3 py-8 text-center text-xs text-muted-foreground">
              {t("common.loading")}
            </div>
          )}

          {!isLoading && list.length === 0 && (
            <div className="px-3 py-8 text-center text-xs text-muted-foreground">
              {t("chat.noConversations")}
            </div>
          )}

          {list.map((conv) => (
            <Link
              key={conv.id}
              to={`/admin/chat/${encodeURIComponent(conv.id)}`}
              onMouseEnter={() => setHoveredId(conv.id)}
              onMouseLeave={() => setHoveredId(null)}
              className={cn(
                "group relative flex w-full items-start gap-2.5 rounded-lg px-3 py-2.5 text-left text-sm transition-colors",
                activeId === conv.id
                  ? "bg-sidebar-accent text-sidebar-foreground"
                  : "hover:bg-secondary text-foreground",
              )}
            >
              {runningId === conv.id ? (
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse shrink-0 mt-1.5" />
              ) : (
                <MessageSquare className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
              )}
              <div className="flex-1 min-w-0">
                {editingId === conv.id ? (
                  <input
                    ref={editRef}
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => {
                      e.stopPropagation();
                      if (e.key === "Enter") { e.preventDefault(); commitRename(); }
                      if (e.key === "Escape") { e.preventDefault(); cancelRename(); }
                    }}
                    onBlur={commitRename}
                    onClick={(e) => e.preventDefault()}
                    className="w-full rounded border border-input bg-background px-1.5 py-0.5 text-[13px] font-medium focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                ) : (
                  <div className="truncate font-medium text-[13px] leading-snug">
                    {conv.title || t("chat.untitled")}
                  </div>
                )}
                <div className="text-[11px] text-muted-foreground mt-0.5">
                  {runningId === conv.id ? (
                    <span>Generating…</span>
                  ) : (
                    <>
                      {formatRelativeTime(conv.updated_at)}
                      {conv.turn_count > 0 && (
                        <span className="ml-2">{conv.turn_count} turns</span>
                      )}
                    </>
                  )}
                </div>
              </div>

              {/* Action buttons — fades in on hover */}
              <div
                className={cn(
                  "absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-0.5 transition-all",
                  hoveredId === conv.id || editingId === conv.id ? "opacity-100" : "opacity-0",
                )}
              >
                {/* Rename */}
                <button
                  type="button"
                  className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    startRename(conv.id, conv.title);
                  }}
                  title="Rename"
                >
                  <Pencil className="w-3 h-3" />
                </button>
                {/* Delete */}
                <button
                  type="button"
                  className="p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setDeleteConfirmId(conv.id);
                  }}
                  title={t("chat.deleteConversation")}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Delete confirmation dialog */}
      <Dialog open={!!deleteConfirmId} onOpenChange={() => setDeleteConfirmId(null)}>
        <DialogContent className="sm:max-w-[380px]">
          <DialogHeader>
            <DialogTitle>{t("chat.deleteConversation")}</DialogTitle>
            <DialogDescription>{t("chat.deleteConfirm")}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setDeleteConfirmId(null)}>
              {t("common.cancel")}
            </Button>
            <Button variant="destructive" size="sm" onClick={handleDeleteConfirm}>
              {t("common.delete")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </aside>
  );
});
