export interface ChatStreamEvent {
  type: "thinking" | "text" | "tool" | "tool_use" | "tool_result" | "status" | "done" | "error";
  content?: string;
  text?: string;
  message?: string;
}

export interface ChatStreamConfig {
  labId: string;
  cellId: string;
  message: string;
  agent: string;
  systemPrompt?: string;
  outputFormat?: string;
  abortSignal?: AbortSignal;
  onEvent: (event: ChatStreamEvent) => void;
  onError: (error: Error) => void;
  onDone: (fullText: string) => void;
}

export async function streamChat(config: ChatStreamConfig): Promise<void> {
  const { labId, cellId, message, agent, systemPrompt, outputFormat, abortSignal, onEvent, onError, onDone } =
    config;

  try {
    const response = await fetch(
      `/admin/api/labs/${encodeURIComponent(labId)}/cells/${encodeURIComponent(cellId)}/chat`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          agent,
          system_prompt: systemPrompt || "",
          output_format: outputFormat || "text",
        }),
        signal: abortSignal,
      },
    );

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

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const event: ChatStreamEvent = JSON.parse(line.slice(6));
            onEvent(event);

            if (event.type === "text" && event.content) {
              fullText += event.content;
            }
            if (event.type === "done") {
              onDone(fullText);
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

    onDone(fullText);
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}
