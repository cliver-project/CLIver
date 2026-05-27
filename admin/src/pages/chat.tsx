import { useState, useCallback, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  type ThreadMessageLike,
  type AppendMessage,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { ArrowUp, Square } from "lucide-react";
import { streamChat, type ChatArtifact } from "@/lib/chat-stream";
import { useConversations, useConversation } from "@/hooks/use-conversations";
import { ConversationSidebar } from "@/components/chat/ConversationSidebar";
import { useTranslation } from "@/i18n";

function generateId(): string {
  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

function convertTurnToMessage(turn: { role: string; content: string }): ThreadMessageLike {
  return {
    id: generateId(),
    role: turn.role as "user" | "assistant",
    content: [{ type: "text" as const, text: turn.content }],
  };
}

export default function ChatPage() {
  const { t } = useTranslation();
  const { conversationId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const activeConversationId = conversationId || null;

  const [messages, setMessages] = useState<ThreadMessageLike[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastArtifacts, setLastArtifacts] = useState<ChatArtifact[]>([]);
  const [artifactMessageId, setArtifactMessageId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const loadedConversationIdRef = useRef<string | null>(null);
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  const { data: conversations, isLoading: convsLoading } = useConversations();
  const { data: conversationDetail } = useConversation(activeConversationId);

  // Load turns when active conversation changes
  useEffect(() => {
    if (isRunning) return;
    if (activeConversationId && conversationDetail?.turns) {
      if (loadedConversationIdRef.current === activeConversationId) return;
      loadedConversationIdRef.current = activeConversationId;
      setMessages(conversationDetail.turns.map(convertTurnToMessage));
    } else if (!activeConversationId) {
      loadedConversationIdRef.current = null;
      setMessages([]);
      setError(null);
    }
  }, [activeConversationId, conversationDetail, isRunning]);

  // Scroll to bottom when messages change (history load or new messages)
  useEffect(() => {
    if (scrollAnchorRef.current) {
      scrollAnchorRef.current.scrollIntoView({ behavior: "instant" as ScrollBehavior });
    }
  }, [messages]);

  // Redirect to most recent conversation if at /chat with no id
  const didInitialRedirect = useRef(false);
  useEffect(() => {
    if (!didInitialRedirect.current && !activeConversationId && conversations && conversations.length > 0) {
      didInitialRedirect.current = true;
      const first = conversations[0];
      if (first) navigate(`/admin/chat/${encodeURIComponent(first.id)}`, { replace: true });
    }
  }, [activeConversationId, conversations, navigate]);

  const handleNewChat = useCallback(() => {
    navigate("/admin/chat");
  }, [navigate]);

  const handleDelete = useCallback(
    (id: string) => {
      if (activeConversationId === id) {
        navigate("/admin/chat", { replace: true });
      }
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
    [activeConversationId, navigate, queryClient],
  );

  const onNew = useCallback(
    async (message: AppendMessage) => {
      const first = message.content?.[0];
      const input =
        first && typeof first !== "string" && first.type === "text" ? first.text : "";
      if (!input) return;

      setError(null);
      setLastArtifacts([]);
      setArtifactMessageId(null);

      const userMsg: ThreadMessageLike = {
        id: generateId(),
        role: "user",
        content: [{ type: "text", text: input }],
      };

      const assistantId = generateId();
      const assistantMsg: ThreadMessageLike = {
        id: assistantId,
        role: "assistant",
        content: [{ type: "text", text: "" }],
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsRunning(true);

      const controller = new AbortController();
      abortRef.current = controller;

      let fullText = "";

      await streamChat({
        message: input,
        conversationId: activeConversationId ?? undefined,
        abortSignal: controller.signal,
        onEvent: (event) => {
          let chunk = "";
          if (event.type === "text" && event.content) {
            chunk = event.content;
          } else if (event.type === "thinking" && event.content) {
            chunk = `\n> ${event.content}\n`;
          } else if (event.type === "tool_use" && event.content) {
            const label =
              event.content.length > 60
                ? event.content.slice(0, 60) + "..."
                : event.content;
            chunk = `\n\`\`\`\n${label}\n\`\`\`\n`;
          } else if (event.type === "tool_result" && event.content) {
            const truncated =
              event.content.length > 600
                ? event.content.slice(0, 600) + "..."
                : event.content;
            chunk = `\n\`\`\`\n${truncated}\n\`\`\`\n`;
          } else if (event.type === "status" && event.content) {
            chunk = `\n_${event.content}_\n`;
          }

          if (chunk) {
            fullText += chunk;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: [{ type: "text" as const, text: fullText }] }
                  : m,
              ),
            );
          }
        },
        onError: (err) => {
          setError(err.message);
          setIsRunning(false);
          abortRef.current = null;
        },
        onDone: (_fullText, artifacts, sessionId) => {
          setIsRunning(false);
          abortRef.current = null;
          if (artifacts.length > 0) {
            setLastArtifacts(artifacts);
            setArtifactMessageId(assistantId);
          }
          if (sessionId && !activeConversationId) {
            loadedConversationIdRef.current = sessionId;
            navigate(`/admin/chat/${encodeURIComponent(sessionId)}`, { replace: true });
          }
          queryClient.invalidateQueries({ queryKey: ["conversations"] });
        },
      });
    },
    [activeConversationId, navigate, queryClient],
  );

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsRunning(false);
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (
        last?.role === "assistant" &&
        last.content?.[0]?.type === "text" &&
        !last.content[0].text
      ) {
        return prev.slice(0, -1);
      }
      return prev;
    });
  }, []);

  const runtime = useExternalStoreRuntime({
    isRunning,
    isSendDisabled: isRunning,
    messages,
    convertMessage: (msg: ThreadMessageLike) => msg,
    onNew,
    onCancel: handleCancel,
  });

  const showEmptyState = !activeConversationId && messages.length === 0;

  return (
    <div className="h-full -m-6 flex overflow-hidden">
      <ConversationSidebar
        conversations={conversations || []}
        activeId={activeConversationId}
        onNew={handleNewChat}
        onDelete={handleDelete}
        isLoading={convsLoading}
      />

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden bg-background">
        {error && (
          <div className="px-4 py-2 text-xs text-red-700 bg-red-50 border-b border-red-200 shrink-0">
            {error}
          </div>
        )}

        <AssistantRuntimeProvider runtime={runtime}>
          <ThreadPrimitive.Root className="flex flex-col h-full">
            <ThreadPrimitive.Viewport
              className="flex-1 min-h-0 overflow-y-auto"
              autoScroll
            >
              <div className="max-w-4xl mx-auto w-full px-4 lg:px-6 py-4 lg:py-6">
                {showEmptyState && (
                  <div className="flex flex-col items-center justify-center text-center min-h-[400px]">
                    <h1 className="text-2xl font-semibold text-foreground mb-2">
                      {t("chat.emptyStateTitle")}
                    </h1>
                    <p className="text-sm text-muted-foreground max-w-md leading-relaxed">
                      {t("chat.emptyStateDesc")}
                    </p>
                  </div>
                )}

                <div className="space-y-4">
                  <ThreadPrimitive.Messages>
                    {({ message }) => (
                      <div
                        className={`message-enter flex ${
                          message.role === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        <div
                          className={`message-bubble ${
                            message.role === "user"
                              ? "message-bubble-user"
                              : "message-bubble-assistant"
                          }`}
                        >
                          <MessagePrimitive.Content
                            components={{
                              Text: ({ text, status }) => {
                                if (status?.type === "running" && !text) {
                                  return (
                                    <div className="loading-dots flex gap-1 py-1">
                                      <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                                      <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                                      <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                                    </div>
                                  );
                                }
                                return <MarkdownTextPrimitive text={text} />;
                              },
                            }}
                          />
                          {message.id === artifactMessageId && lastArtifacts.length > 0 && (
                            <div className="mt-2 pt-2 border-t space-y-2">
                              {lastArtifacts.map((a, i) => (
                                <ChatArtifactPreview key={i} artifact={a} />
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </ThreadPrimitive.Messages>

                  {/* Loading indicator */}
                  {isRunning && messages[messages.length - 1]?.role === "assistant" && (
                    (() => {
                      const last = messages[messages.length - 1];
                      const text = last.content?.[0]?.type === "text"
                        ? (last.content[0] as { text: string }).text
                        : "";
                      if (text) return null;
                      return (
                        <div className="flex justify-start message-enter">
                          <div className="message-bubble message-bubble-assistant">
                            <div className="loading-dots flex gap-1">
                              <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                              <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                              <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                            </div>
                          </div>
                        </div>
                      );
                    })()
                  )}
                  <div ref={scrollAnchorRef} />
                </div>
              </div>
            </ThreadPrimitive.Viewport>

            {/* Composer */}
            <div className="border-t border-border bg-background shrink-0 p-4 lg:p-6">
              <div className="max-w-4xl mx-auto w-full">
                <ComposerPrimitive.Root>
                  <div className="flex gap-3 items-end">
                    <ComposerPrimitive.Input
                      className="flex-1 min-w-0 rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground disabled:opacity-50"
                      placeholder={t("chat.typeMessage")}
                    />
                    {isRunning ? (
                      <button
                        type="button"
                        className="inline-flex items-center justify-center rounded-lg p-2.5 text-sm font-medium border border-input hover:bg-muted transition-colors shrink-0"
                        onClick={handleCancel}
                      >
                        <Square className="w-4 h-4" />
                      </button>
                    ) : (
                      <ComposerPrimitive.Send asChild>
                        <button
                          type="button"
                          className="inline-flex items-center justify-center rounded-lg p-2.5 text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors shrink-0 disabled:opacity-50"
                        >
                          <ArrowUp className="w-4 h-4" />
                        </button>
                      </ComposerPrimitive.Send>
                    )}
                  </div>
                </ComposerPrimitive.Root>
              </div>
            </div>
          </ThreadPrimitive.Root>
        </AssistantRuntimeProvider>
      </main>
    </div>
  );
}

function ChatArtifactPreview({ artifact }: { artifact: ChatArtifact }) {
  const mediaType = artifact.media_type;
  const name = artifact.path.split("/").pop() || artifact.path;

  if (mediaType.startsWith("image/")) {
    return (
      <div className="rounded-md overflow-hidden border">
        <img
          src={`/admin/api/files?path=${encodeURIComponent(artifact.path)}`}
          alt={name}
          className="max-w-full h-auto max-h-64 object-contain"
        />
        <div className="px-2 py-1 text-[10px] text-muted-foreground bg-muted/50">
          {name} {artifact.size ? `(${(artifact.size / 1024).toFixed(1)} KB)` : ""}
        </div>
      </div>
    );
  }

  if (mediaType.startsWith("audio/")) {
    return (
      <div className="rounded-md border p-2">
        <audio controls className="w-full h-8">
          <source src={`/admin/api/files?path=${encodeURIComponent(artifact.path)}`} type={mediaType} />
        </audio>
        <div className="text-[10px] text-muted-foreground mt-1">{name}</div>
      </div>
    );
  }

  if (mediaType.startsWith("video/")) {
    return (
      <div className="rounded-md overflow-hidden border">
        <video controls className="max-w-full max-h-48">
          <source src={`/admin/api/files?path=${encodeURIComponent(artifact.path)}`} type={mediaType} />
        </video>
        <div className="px-2 py-1 text-[10px] text-muted-foreground bg-muted/50">{name}</div>
      </div>
    );
  }

  return (
    <a
      href={`/admin/api/files?path=${encodeURIComponent(artifact.path)}`}
      className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-muted/50 transition-colors"
      download={name}
    >
      <span className="text-muted-foreground">📄</span>
      <span className="flex-1 truncate">{name}</span>
      {artifact.size && (
        <span className="text-[10px] text-muted-foreground">
          {(artifact.size / 1024).toFixed(1)} KB
        </span>
      )}
    </a>
  );
}
