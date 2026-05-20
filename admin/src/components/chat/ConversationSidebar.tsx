import { useState } from "react";
import { Plus, Trash2, MessageSquare } from "lucide-react";
import { useTranslation } from "@/i18n";
import { cn } from "@/lib/utils";
import type { Conversation } from "@/hooks/use-conversations";

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
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  isLoading: boolean;
}

export function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  isLoading,
}: ConversationSidebarProps) {
  const { t } = useTranslation();
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  return (
    <aside className="w-[260px] flex flex-col border-r border-sidebar-border bg-sidebar-background shrink-0">
      {/* New Chat button */}
      <div className="p-3 border-b border-sidebar-border">
        <button
          type="button"
          onClick={onNew}
          className="flex items-center justify-center gap-2 w-full rounded-lg border border-border px-3 py-2 text-sm font-medium hover:bg-secondary transition-colors"
        >
          <Plus className="w-4 h-4" />
          {t("chat.newChat")}
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-1">
        {isLoading && conversations.length === 0 && (
          <div className="px-3 py-8 text-center text-xs text-muted-foreground">
            {t("common.loading")}
          </div>
        )}

        {!isLoading && conversations.length === 0 && (
          <div className="px-3 py-8 text-center text-xs text-muted-foreground">
            {t("chat.noConversations")}
          </div>
        )}

        {conversations.map((conv) => (
          <div
            key={conv.id}
            role="button"
            tabIndex={0}
            onClick={() => onSelect(conv.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(conv.id);
              }
            }}
            onMouseEnter={() => setHoveredId(conv.id)}
            onMouseLeave={() => setHoveredId(null)}
            className={cn(
              "group relative flex w-full items-start gap-2.5 rounded-lg px-3 py-2.5 text-left text-sm transition-colors cursor-pointer",
              activeId === conv.id
                ? "bg-sidebar-accent text-sidebar-foreground"
                : "hover:bg-secondary text-foreground",
            )}
          >
            <MessageSquare className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
            <div className="flex-1 min-w-0">
              <div className="truncate font-medium text-[13px] leading-snug">
                {conv.title || t("chat.untitled")}
              </div>
              <div className="text-[11px] text-muted-foreground mt-0.5">
                {formatRelativeTime(conv.updated_at)}
                {conv.turn_count > 0 && (
                  <span className="ml-2">{conv.turn_count} turns</span>
                )}
              </div>
            </div>

            {/* Delete — fades in on hover */}
            <button
              type="button"
              className={cn(
                "absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded transition-all",
                "hover:bg-destructive/10 text-muted-foreground hover:text-destructive",
                hoveredId === conv.id ? "opacity-100" : "opacity-0",
              )}
              onClick={(e) => {
                e.stopPropagation();
                onDelete(conv.id);
              }}
              title={t("chat.deleteConversation")}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
