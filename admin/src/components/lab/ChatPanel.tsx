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
import { streamChat } from "@/lib/chat-stream";
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
  onSaveResult: (text: string) => void;
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
  onSaveResult,
}: ChatPanelProps) {
  const { t } = useTranslation();
  const [messages, setMessages] = useState<ThreadMessageLike[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [lastAssistantText, setLastAssistantText] = useState("");
  const [error, setError] = useState<string | null>(null);
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
        onDone: () => {
          setLastAssistantText(fullText);
          setIsRunning(false);
          abortRef.current = null;
        },
      });
    },
    [labId, cellId, agent, systemPrompt, outputFormat],
  );

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

    try {
      const res = await fetch(
        `/admin/api/labs/${encodeURIComponent(labId)}/cells/${encodeURIComponent(cellId)}/save-output`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ outputs: { text } }),
        },
      );
      if (!res.ok) {
        setError(`Failed to save: ${res.status}`);
        return;
      }
      onSaveResult(text);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save result");
    }
  }, [labId, cellId, lastAssistantText, onSaveResult]);

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
                className="w-full gap-1.5"
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
