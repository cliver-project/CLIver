import { useState, useCallback, useEffect, useRef } from "react";
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
import { Button } from "@/components/ui/button";
import { Save } from "lucide-react";
import { streamChat, type ChatArtifact } from "@/lib/chat-stream";
import { useTranslation } from "@/i18n";

interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
}

interface ChatPanelProps {
  labId: string;
  cellId: string;
  agent: string;
  systemPrompt: string;
  outputFormat: string;
  initialPrompt?: string;
  runTrigger?: number;
  onSaveResult: (text: string, artifacts: ChatArtifact[]) => void;
}

function generateId(): string {
  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

function convertTurnToMessage(turn: ChatTurn): ThreadMessageLike {
  return {
    id: generateId(),
    role: turn.role,
    content: [{ type: "text" as const, text: turn.content }],
  };
}

export function ChatPanel({
  labId,
  cellId,
  agent,
  systemPrompt,
  outputFormat,
  initialPrompt,
  runTrigger,
  onSaveResult,
}: ChatPanelProps) {
  const { t } = useTranslation();
  const [messages, setMessages] = useState<ThreadMessageLike[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [lastAssistantText, setLastAssistantText] = useState("");
  const [lastArtifacts, setLastArtifacts] = useState<ChatArtifact[]>([]);
  const [artifactMessageId, setArtifactMessageId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Load chat history on mount
  useEffect(() => {
    let cancelled = false;
    async function loadHistory() {
      try {
        const res = await fetch(
          `/admin/api/labs/${encodeURIComponent(labId)}/cells/${encodeURIComponent(cellId)}/chat/history`,
          { credentials: "include" },
        );
        if (!res.ok || cancelled) return;
        const data = await res.json();
        if (data.turns && data.turns.length > 0) {
          const msgs = data.turns.map(convertTurnToMessage);
          if (!cancelled) setMessages(msgs);
        }
      } catch {
        // Non-critical — start with empty history
      }
      if (!cancelled) setHistoryLoaded(true);
    }
    loadHistory();
    return () => {
      cancelled = true;
    };
  }, [labId, cellId]);

  const onNew = useCallback(
    async (message: AppendMessage) => {
      const input =
        message.content?.[0]?.type === "text" ? message.content[0].text : "";
      if (!input) return;

      setError(null);
      setLastAssistantText("");
      setLastArtifacts([]);
      setArtifactMessageId(null);

      // Create user message and empty assistant message for streaming
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
        labId,
        cellId,
        message: input,
        agent,
        systemPrompt,
        outputFormat,
        abortSignal: controller.signal,
        onEvent: (event) => {
          if (event.type === "text" && event.content) {
            fullText += event.content;
            // Update the assistant message progressively for streaming
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: [{ type: "text" as const, text: fullText }],
                    }
                  : m,
              ),
            );
          }
        },
        onError: (err) => {
          setError(err.message);
          setLastAssistantText("");
          setIsRunning(false);
          abortRef.current = null;
        },
        onDone: (finalText: string, artifacts: ChatArtifact[]) => {
          setLastAssistantText(finalText);
          if (artifacts.length > 0) {
            setLastArtifacts(artifacts);
            setArtifactMessageId(assistantId);
          }
          setIsRunning(false);
          abortRef.current = null;
        },
      });
    },
    [labId, cellId, agent, systemPrompt, outputFormat],
  );

  // Ref to always access the latest onNew without stale closures
  const onNewRef = useRef(onNew);
  onNewRef.current = onNew;

  // When runTrigger increments, send the initial prompt
  useEffect(() => {
    if (runTrigger && runTrigger > 0 && initialPrompt && messages.length === 0) {
      onNewRef.current({
        content: [{ type: "text", text: initialPrompt }],
      });
    }
    // We intentionally only react to runTrigger changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runTrigger]);

  const handleCancel = useCallback(async () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsRunning(false);
    // Remove the empty assistant message
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "assistant" && last.content?.[0]?.type === "text" && !last.content[0].text) {
        return prev.slice(0, -1);
      }
      return prev;
    });
  }, []);

  const handleSave = useCallback(async () => {
    const text = lastAssistantText;
    if (!text) return;

    const outputs: Record<string, unknown> = { text };
    if (lastArtifacts.length > 0) {
      outputs.artifacts = lastArtifacts;
    }

    try {
      const res = await fetch(
        `/admin/api/labs/${encodeURIComponent(labId)}/cells/${encodeURIComponent(cellId)}/save-output`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ outputs }),
        },
      );
      if (!res.ok) {
        setError(`Failed to save: ${res.status}`);
        return;
      }
      onSaveResult(text, lastArtifacts);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save result");
    }
  }, [labId, cellId, lastAssistantText, lastArtifacts, onSaveResult]);

  const runtime = useExternalStoreRuntime({
    isRunning,
    isSendDisabled: isRunning,
    messages,
    convertMessage: (msg: ThreadMessageLike) => msg,
    onNew,
    onCancel: handleCancel,
  });

  return (
    <div className="flex flex-col h-full">
      {error && (
        <div className="px-3 py-2 text-xs text-red-700 bg-red-50 border-b border-red-200">
          {error}
        </div>
      )}

      <AssistantRuntimeProvider runtime={runtime}>
        <ThreadPrimitive.Root className="flex flex-col h-full">
          <ThreadPrimitive.Viewport
            className="flex-1 min-h-0 overflow-y-auto px-4 py-3"
            autoScroll
          >
            {/* Initial prompt indicator — shown before Run is clicked */}
            {initialPrompt && historyLoaded && messages.length === 0 && (
              <div className="mb-4 rounded-lg border-2 border-dashed border-primary/30 bg-primary/5 px-4 py-3">
                <div className="text-[10px] font-semibold text-primary/60 uppercase tracking-wider mb-1">
                  {t("lab.initialPrompt")}
                </div>
                <div className="text-sm text-foreground whitespace-pre-wrap">
                  {initialPrompt}
                </div>
              </div>
            )}

            <ThreadPrimitive.Messages>
              {({ message }) => (
                <div className="mb-3 flex">
                  <div
                    className={`rounded-lg px-3 py-2 text-sm max-w-[80%] ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground ml-auto"
                        : "bg-muted"
                    }`}
                  >
                    <MessagePrimitive.Content
                      components={{
                        Text: ({ text, status }) => {
                          if (status?.type === "running" && !text) {
                            return (
                              <span className="italic text-muted-foreground">
                                {t("lab.generating")}
                              </span>
                            );
                          }
                          return <MarkdownTextPrimitive text={text} />;
                        },
                      }}
                    />
                    {message.id === artifactMessageId && lastArtifacts.length > 0 && (
                      <div className="mt-2 pt-2 border-t space-y-2">
                        {lastArtifacts.map((a, i) => (
                          <ArtifactPreview key={i} artifact={a} />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </ThreadPrimitive.Messages>
          </ThreadPrimitive.Viewport>

          <div className="border-t px-4 py-3 space-y-2">
            <ComposerPrimitive.Root>
              <div className="flex gap-2">
                <ComposerPrimitive.Input
                  className="flex-1 min-w-0"
                  placeholder="Type a message..."
                />
                {isRunning ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCancel}
                  >
                    {t("common.cancel")}
                  </Button>
                ) : (
                  <ComposerPrimitive.Send asChild>
                    <Button variant="default" size="sm">
                      Send
                    </Button>
                  </ComposerPrimitive.Send>
                )}
              </div>
            </ComposerPrimitive.Root>

            {lastAssistantText && !isRunning && (
              <Button
                variant="outline"
                size="sm"
                className="gap-1.5"
                onClick={handleSave}
              >
                <Save className="w-3.5 h-3.5" />
                <span className="text-xs">{t("lab.saveAsResult")}</span>
              </Button>
            )}
          </div>
        </ThreadPrimitive.Root>
      </AssistantRuntimeProvider>
    </div>
  );
}

function ArtifactPreview({ artifact }: { artifact: ChatArtifact }) {
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
