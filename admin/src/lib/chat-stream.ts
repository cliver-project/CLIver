export interface ChatArtifact {
  path: string;
  media_type: string;
  size?: number;
}

export interface ChatStreamEvent {
  type: "thinking" | "text" | "tool" | "tool_use" | "tool_result" | "status" | "done" | "error";
  content?: string;
  text?: string;
  message?: string;
  data?: unknown;
  artifacts?: ChatArtifact[];
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  reasoning_content?: string;
}

export interface ChatStreamConfig {
  // Lab mode: set both labId and cellId to use lab-cell chat endpoint.
  // General mode: omit both to use the central /admin/api/chat endpoint.
  labId?: string;
  cellId?: string;

  // Common
  message: string;
  abortSignal?: AbortSignal;
  onEvent: (event: ChatStreamEvent) => void;
  onError: (error: Error) => void;
  onDone: (fullText: string, artifacts: ChatArtifact[], sessionId?: string) => void;

  // Lab-specific (used only when labId + cellId are provided)
  agent?: string;
  systemPrompt?: string;
  outputFormat?: string;

  // General-chat params (used only when labId + cellId are omitted)
  agent?: string;
  model?: string;
  systemMessage?: string;
  conversationHistory?: ConversationMessage[];
  conversationId?: string;
  filterTools?: string[];
  saveMediaDir?: string;
}

function isLabMode(config: ChatStreamConfig): config is ChatStreamConfig & { labId: string; cellId: string } {
  return !!(config.labId && config.cellId);
}

export async function streamChat(config: ChatStreamConfig): Promise<void> {
  const { message, abortSignal, onEvent, onError, onDone } = config;

  const url = isLabMode(config)
    ? `/admin/api/labs/${encodeURIComponent(config.labId)}/cells/${encodeURIComponent(config.cellId)}/chat`
    : "/admin/api/chat";

  const body = isLabMode(config)
    ? JSON.stringify({
        message,
        agent: config.agent,
        system_prompt: config.systemPrompt || "",
        output_format: config.outputFormat || "text",
      })
    : JSON.stringify({
        message,
        agent: config.agent,
        model: config.model,
        system_message: config.systemMessage,
        conversation_history: config.conversationHistory,
        session_id: config.conversationId,
        filter_tools: config.filterTools,
        save_media_dir: config.saveMediaDir,
      });

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: abortSignal,
    });

    if (!response.ok) {
      const text = await response.text();
      onError(new Error(`Chat error ${response.status}: ${text}`));
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      onError(new Error("No response body"));
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";
    let artifacts: ChatArtifact[] = [];

    // General chat emits "chunk"/"content" events; normalize to "text" for callers.
    const normalizeEvent = (raw: Record<string, unknown>): ChatStreamEvent => {
      if (raw.type === "chunk" || raw.type === "content") {
        return { type: "text", content: raw.content as string };
      }
      return raw as unknown as ChatStreamEvent;
    };

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
            const event = normalizeEvent(raw);

            onEvent(event);

            if (event.type === "text" && event.content) {
              fullText += event.content;
            }
            if (event.type === "done") {
              // General chat returns media/media_files, lab chat returns artifacts
              const rawArtifacts = (raw.media || raw.artifacts || raw.media_files) as ChatArtifact[] | undefined;
              if (rawArtifacts) {
                artifacts = rawArtifacts;
              }
              const sessionId = raw.session_id as string | undefined;
              onDone(fullText || (event.text || event.content as string) || "", artifacts, sessionId);
              return;
            }
            if (event.type === "error") {
              onError(new Error(event.message || "Unknown error"));
              return;
            }
          } catch {
            // Skip unparseable lines
          }
        }
      }
    }

    onDone(fullText, artifacts);
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}
