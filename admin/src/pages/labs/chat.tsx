import { useState, useCallback, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  ThreadPrimitive,
  MessagePrimitive,
  type ThreadMessageLike,
  type AppendMessage,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { ArrowUp, Square, Play, FileText, Brain, Package } from "lucide-react";
import { useLab, useLabGoldenTests, useRunGoldenTests, type TestRunResult } from "@/hooks/use-api";
import { LabHeader } from "@/components/lab/LabHeader";
import { LabConfigPanel } from "@/components/lab/LabConfigPanel";
import { GoldenTestCard } from "@/components/lab/GoldenTestCard";
import { useTranslation } from "@/i18n";

function generateId(): string {
  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

export default function LabChatPage() {
  const { t } = useTranslation();
  const { labId, sessionId: paramSessionId } = useParams<{ labId: string; sessionId?: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: labDetail, isLoading: labLoading } = useLab(labId);
  const { data: tests } = useLabGoldenTests(labId);
  const runTests = useRunGoldenTests(labId!);

  const activeSessionId = paramSessionId || null;

  const [messagesByConv, setMessagesByConv] = useState<Record<string, ThreadMessageLike[]>>({});
  const [runningConvId, setRunningConvId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const loadedSessionIds = useRef<Set<string>>(new Set());
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  const messages = activeSessionId ? (messagesByConv[activeSessionId] || []) : [];
  const isRunning = !!(runningConvId && runningConvId === activeSessionId);

  const [inputText, setInputText] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [savingConfig, setSavingConfig] = useState(false);
  const [testResults, setTestResults] = useState<TestRunResult[] | null>(null);

  useEffect(() => {
    if (!activeSessionId) {
      setSelectedModel("");
      setSystemPrompt("");
      setSelectedSkills([]);
      return;
    }
    fetch(`/admin/api/conversations/${encodeURIComponent(activeSessionId)}`, { credentials: "include" })
      .then((r) => r.json())
      .then((data) => {
        const opts = (data?.session?.options as Record<string, unknown>) || {};
        if (opts.model) setSelectedModel(String(opts.model));
        if (opts.system_prompt) setSystemPrompt(String(opts.system_prompt));
        if (opts.skills) setSelectedSkills(opts.skills as string[]);
      })
      .catch(() => {});
  }, [activeSessionId]);

  useEffect(() => {
    if (scrollAnchorRef.current) {
      scrollAnchorRef.current.scrollIntoView({ behavior: "instant" as ScrollBehavior });
    }
  }, [messages]);

  const handleSaveConfig = useCallback(async () => {
    if (!activeSessionId) return;
    setSavingConfig(true);
    try {
      await fetch(
        `/admin/api/labs/${encodeURIComponent(labId!)}/chat/${encodeURIComponent(activeSessionId)}`,
        {
          method: "PATCH",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            options: {
              model: selectedModel || null,
              system_prompt: systemPrompt || null,
              skills: selectedSkills.length > 0 ? selectedSkills : null,
            },
          }),
        },
      );
    } catch {}
    setSavingConfig(false);
  }, [activeSessionId, labId, selectedModel, systemPrompt, selectedSkills]);

  const handleSend = useCallback(() => {
    const text = inputText.trim();
    if (!text || isRunning) return;
    onNewRef.current?.({
      role: "user",
      content: [{ type: "text" as const, text }],
      parentId: null,
      sourceId: null,
      runConfig: undefined,
    } as AppendMessage);
    setInputText("");
  }, [inputText, isRunning]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      if (e.nativeEvent.isComposing) return;
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunningConvId(null);
  }, []);

  const onNew = useCallback(
    async (message: AppendMessage) => {
      const first = message.content?.[0];
      const input = first && typeof first !== "string" && first.type === "text" ? first.text : "";
      if (!input) return;

      const convId = activeSessionId;
      const convLabId = labId!;

      const updateConvMessages = (updater: (prev: ThreadMessageLike[]) => ThreadMessageLike[]) => {
        setMessagesByConv((prev) => ({
          ...prev,
          [convId!]: updater(prev[convId!] || []),
        }));
      };

      setError(null);
      setTestResults(null);

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

      updateConvMessages((prev) => [...prev, userMsg, assistantMsg]);
      setRunningConvId(convId);

      const controller = new AbortController();
      abortRef.current = controller;

      let fullText = "";

      try {
        const body = JSON.stringify({
          message: input,
          conversation_id: convId ?? undefined,
          model: selectedModel || undefined,
          system_message: systemPrompt || undefined,
          filter_tools: selectedSkills.length > 0 ? selectedSkills : undefined,
        });

        const response = await fetch(
          `/admin/api/labs/${encodeURIComponent(convLabId)}/chat${convId ? `/${encodeURIComponent(convId)}` : ""}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
            signal: controller.signal,
            credentials: "include",
          },
        );

        if (!response.ok) {
          const text = await response.text();
          setError(`Error ${response.status}: ${text}`);
          setRunningConvId(null);
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) { setRunningConvId(null); return; }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const raw = JSON.parse(line.slice(6));
                if (raw.type === "session" && raw.session_id) {
                  loadedSessionIds.current.add(raw.session_id);
                  if (!convId) {
                    navigate(`/admin/labs/${convLabId}/chat/${raw.session_id}`, { replace: true });
                  }
                  continue;
                }
                if (raw.type === "content" && raw.content) {
                  fullText += raw.content;
                  updateConvMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId
                        ? { ...m, content: [{ type: "text" as const, text: fullText }] }
                        : m,
                    ),
                  );
                }
                if (raw.type === "done") {
                  setRunningConvId(null);
                  queryClient.invalidateQueries({ queryKey: ["conversations"] });
                  if (raw.session_id && !convId) {
                    navigate(`/admin/labs/${convLabId}/chat/${raw.session_id}`, { replace: true });
                  }
                }
                if (raw.type === "error") {
                  setError(raw.message || "Unknown error");
                  setRunningConvId(null);
                }
              } catch {}
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof Error ? err.message : String(err));
        setRunningConvId(null);
      }
    },
    [activeSessionId, labId, navigate, queryClient, selectedModel, systemPrompt, selectedSkills],
  );

  const onNewRef = useRef(onNew);
  onNewRef.current = onNew;

  const runtime = useExternalStoreRuntime({
    isRunning,
    isSendDisabled: isRunning,
    messages,
    convertMessage: (msg: ThreadMessageLike) => msg,
    onNew,
    onCancel: handleCancel,
  });

  if (labLoading) {
    return <p className="text-sm text-muted-foreground p-4">{t("common.loading")}</p>;
  }

  if (!labDetail) {
    return <p className="text-sm text-muted-foreground p-4">{t("labs.notFound")}</p>;
  }

  const lab = labDetail.lab;
  const showEmptyState = !activeSessionId && messages.length === 0;

  return (
    <div className="flex flex-col h-full -m-6">
      <LabHeader
        title={lab.title}
        description={lab.description}
        breadcrumb={
          <>
            <button onClick={() => navigate("/admin/labs")} className="hover:text-foreground transition-colors">
              {t("labs.title")}
            </button>
            <span className="mx-0.5">›</span>
            <button onClick={() => navigate(`/admin/labs/${labId}`)} className="hover:text-foreground transition-colors">
              {lab.title}
            </button>
          </>
        }
      />

      <div className="flex flex-1 min-h-0">
        {/* LEFT PANEL */}
        <div className="w-[240px] min-w-[240px] border-r bg-card overflow-hidden">
          <LabConfigPanel
            selectedModel={selectedModel}
            systemPrompt={systemPrompt}
            selectedSkills={selectedSkills}
            onModelChange={setSelectedModel}
            onSystemPromptChange={setSystemPrompt}
            onSkillsChange={setSelectedSkills}
            onSave={handleSaveConfig}
            saving={savingConfig}
          />
        </div>

        {/* MAIN PANEL */}
        <div className="flex-1 flex flex-col min-w-0 bg-background">
          {error && (
            <div className="px-4 py-2 text-xs text-red-700 bg-red-50 border-b border-red-200 shrink-0">
              {error}
            </div>
          )}
          <AssistantRuntimeProvider runtime={runtime}>
            <ThreadPrimitive.Root className="grid grid-rows-[1fr_auto] flex-1 min-h-0">
              <ThreadPrimitive.Viewport className="overflow-y-auto min-h-0" autoScroll>
                <div className="max-w-4xl mx-auto w-full px-4 lg:px-6 py-4 lg:py-6">
                {showEmptyState && (
                  <div className="flex flex-col items-center justify-center text-center min-h-[400px]">
                    <h1 className="text-2xl font-semibold text-foreground mb-2">
                      {lab.title}
                    </h1>
                    <p className="text-sm text-muted-foreground max-w-md leading-relaxed">
                      {lab.description || t("lab.chat")}
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
                          </div>
                        </div>
                      )}
                    </ThreadPrimitive.Messages>
                    <div ref={scrollAnchorRef} />
                  </div>
                </div>
              </ThreadPrimitive.Viewport>

              {/* Input box */}
              <div className="border-t bg-card shrink-0 px-4 py-3">
                <div className="max-w-3xl mx-auto w-full flex gap-2 items-end">
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={t("lab.typeMessage")}
                    disabled={isRunning}
                    rows={1}
                    className="flex-1 min-w-0 rounded-lg border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground disabled:opacity-50 resize-none"
                    style={{ minHeight: "40px", maxHeight: "160px" }}
                    onInput={(e) => {
                      const el = e.currentTarget;
                      el.style.height = "auto";
                      el.style.height = Math.min(el.scrollHeight, 160) + "px";
                    }}
                  />
                  {isRunning ? (
                    <button
                      type="button"
                      onClick={handleCancel}
                      className="shrink-0 rounded-lg p-2 border border-input hover:bg-muted transition-colors"
                    >
                      <Square className="w-4 h-4" />
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSend}
                      disabled={!inputText.trim()}
                      className="shrink-0 rounded-lg p-2 bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
                    >
                      <ArrowUp className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            </ThreadPrimitive.Root>
          </AssistantRuntimeProvider>
        </div>

        {/* RIGHT PANEL */}
        <div className="w-[180px] min-w-[180px] border-l bg-card overflow-y-auto p-3 space-y-3">
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
              {t("lab.actions")}
            </p>
            <button
              type="button"
              onClick={async () => {
                const result = await runTests.mutateAsync();
                setTestResults(result.results);
              }}
              disabled={runTests.isPending || !tests || tests.length === 0}
              className="w-full text-left rounded-md px-2 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5" />
              {t("lab.runTests")}
            </button>
            <button
              type="button"
              className="w-full text-left rounded-md px-2 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 text-muted-foreground"
            >
              <FileText className="w-3.5 h-3.5" />
              {t("lab.viewReport")}
            </button>
            <button
              type="button"
              className="w-full text-left rounded-md px-2 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 text-muted-foreground"
              title={t("lab.comingSoon")}
            >
              <Brain className="w-3.5 h-3.5" />
              {t("lab.distillSkill")}
            </button>
            <button
              type="button"
              className="w-full text-left rounded-md px-2 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 text-muted-foreground"
              title={t("lab.comingSoon")}
            >
              <Package className="w-3.5 h-3.5" />
              {t("lab.generatePackage")}
            </button>
          </div>

          {tests && tests.length > 0 && (
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
                {t("lab.goldenTests")} ({tests.length})
              </p>
              <div className="space-y-1.5">
                {tests.map((test) => (
                  <GoldenTestCard
                    key={test.id}
                    name={test.name}
                    input={test.input}
                    expectedOutput={test.expected_output}
                    result={testResults?.find((r) => r.test_id === test.id)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
