import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, apiPost, apiPut, apiDelete } from "@/lib/api";

// --- Status ---
export function useStatus() {
  return useQuery({
    queryKey: ["status"],
    queryFn: () => api<Record<string, unknown>>("/status"),
    refetchInterval: 10_000,
  });
}

// --- Keys ---
export function useKeys() {
  return useQuery({
    queryKey: ["keys"],
    queryFn: () => api<Array<{name: string; description: string; created_at: string; updated_at: string}>>("/keys"),
  });
}

export function useCreateKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {name: string; value: string; description?: string}) =>
      apiPost("/keys", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["keys"] });
    },
  });
}

export function useDeleteKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => apiDelete(`/keys/${encodeURIComponent(name)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["keys"] });
    },
  });
}

// --- Tasks ---
export function useTasks() {
  return useQuery({
    queryKey: ["tasks"],
    queryFn: () => api<Array<Record<string, unknown>>>("/tasks"),
  });
}

export function useTask(name: string) {
  return useQuery({
    queryKey: ["task", name],
    queryFn: () => api<Record<string, unknown>>(`/tasks/${encodeURIComponent(name)}`),
    enabled: !!name,
  });
}

export function useRunTask(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiPost(`/tasks/${encodeURIComponent(name)}/run`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["task", name] });
      qc.invalidateQueries({ queryKey: ["tasks"] });
    },
  });
}

export function useDeleteTask(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiDelete(`/tasks/${encodeURIComponent(name)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["tasks"] }),
  });
}

export function useCreateTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Record<string, unknown>) => apiPost<Record<string, unknown>>("/tasks", data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["tasks"] }),
  });
}

export function useUpdateTask(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Record<string, unknown>) =>
      apiPut<Record<string, unknown>>(`/tasks/${encodeURIComponent(name)}`, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["task", name] });
      qc.invalidateQueries({ queryKey: ["tasks"] });
    },
  });
}

// --- Sessions ---
export function useSessions(source: "cli" | "gateway") {
  return useQuery({
    queryKey: ["sessions", source],
    queryFn: () => api<Array<Record<string, unknown>>>(`/sessions/${source}`),
  });
}

export function useSessionTurns(source: string, sessionId: string) {
  return useQuery({
    queryKey: ["session-turns", source, sessionId],
    queryFn: () =>
      api<Array<{ role: string; content: string; timestamp?: string }>>(
        `/sessions/${source}/${encodeURIComponent(sessionId)}`,
      ),
    enabled: !!sessionId,
  });
}

export function useDeleteSession(source: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) =>
      apiDelete(`/sessions/${source}/${encodeURIComponent(sessionId)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions", source] }),
  });
}

// --- Skills ---
export function useSkills() {
  return useQuery({
    queryKey: ["skills"],
    queryFn: () => api<Array<Record<string, unknown>>>("/skills"),
  });
}

export function useSkill(name: string) {
  return useQuery({
    queryKey: ["skill", name],
    queryFn: () => api<Record<string, unknown>>(`/skills/${encodeURIComponent(name)}`),
    enabled: !!name,
  });
}

export function useCreateSkill() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; content: string }) =>
      apiPost<{ ok: boolean; name: string }>("/skills", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["skills"] });
    },
  });
}

export function useSaveSkill(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { content: string; path: string }) =>
      apiPut(`/skills/${encodeURIComponent(name)}`, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["skill", name] });
      qc.invalidateQueries({ queryKey: ["skills"] });
    },
  });
}

// --- Agents ---
export function useAgents() {
  return useQuery({
    queryKey: ["agents"],
    queryFn: () => api<Array<Record<string, unknown>>>("/agents"),
  });
}

// --- Config ---
export function useConfig() {
  return useQuery({
    queryKey: ["config"],
    queryFn: () => api<Record<string, unknown>>("/config"),
  });
}

export function useSaveConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Record<string, unknown>) => apiPut("/config", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["config"] });
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

// --- Gateway Restart ---
export function useRestartGateway() {
  return useMutation({
    mutationFn: () => apiPost<{ status: string }>("/restart"),
  });
}

// --- Provider Test ---
export function useTestProvider(name: string) {
  return useMutation({
    mutationFn: (model?: string) =>
      apiPost<{ status: string; message?: string; error?: string }>(
        `/providers/${encodeURIComponent(name)}/test`,
        model ? { model } : undefined,
      ),
  });
}

// --- Models ---
export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: () => api<{ models: string[]; default: string }>("/models"),
  });
}

// --- Templates ---
export interface ChatTemplate {
  id: string;
  label: string;
  system_prompt: string;
  skills: string[];
  agent?: string | null;
  knowledge_base?: string | null;
  description?: string | null;
}

export function useTemplates() {
  return useQuery({
    queryKey: ["templates"],
    queryFn: () => api<ChatTemplate[]>("/templates"),
  });
}

// --- Adapters ---
export function useAdapters() {
  return useQuery({
    queryKey: ["adapters"],
    queryFn: () => api<Array<Record<string, unknown>>>("/adapters"),
  });
}

// --- Scenarios ---
export function useScenarios() {
  return useQuery({
    queryKey: ["scenarios"],
    queryFn: () => api<Array<Record<string, unknown>>>("/scenarios"),
  });
}

export function useInstallScenario() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (source: string) => apiPost("/scenarios/install", { source }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["scenarios"] });
    },
  });
}

export function useRemoveScenario() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/scenarios/${encodeURIComponent(id)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["scenarios"] });
    },
  });
}
