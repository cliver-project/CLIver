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

// --- Workflows ---
export function useWorkflows() {
  return useQuery({
    queryKey: ["workflows"],
    queryFn: () =>
      api<
        Array<{
          name: string;
          description?: string;
          steps: number;
          source?: string;
        }>
      >("/workflows"),
  });
}

export function useWorkflow(name: string) {
  return useQuery({
    queryKey: ["workflow", name],
    queryFn: () =>
      api<{ workflow: Record<string, unknown>; models: string[] }>(
        `/workflows/${encodeURIComponent(name)}`,
      ),
    enabled: !!name,
  });
}

export function useSaveWorkflow(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Record<string, unknown>) =>
      apiPut(`/workflows/${encodeURIComponent(name)}`, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["workflow", name] });
      qc.invalidateQueries({ queryKey: ["workflows"] });
    },
  });
}

export function useRunWorkflow(name: string) {
  return useMutation({
    mutationFn: (inputs?: Record<string, string>) =>
      apiPost(
        `/workflows/${encodeURIComponent(name)}/run`,
        inputs ? { inputs } : undefined,
      ),
  });
}

export function useExecutions(name?: string) {
  return useQuery({
    queryKey: ["executions", name],
    queryFn: () =>
      name
        ? api<Array<Record<string, unknown>>>(
            `/workflows/${encodeURIComponent(name)}/executions`,
          )
        : api<Array<Record<string, unknown>>>("/workflow-executions"),
  });
}

export function useExecutionStatus(name: string, executionId: string) {
  return useQuery({
    queryKey: ["execution-status", name, executionId],
    queryFn: () =>
      api<Record<string, unknown>>(
        `/workflows/${encodeURIComponent(name)}/executions/${encodeURIComponent(executionId)}`,
      ),
    enabled: !!executionId,
    refetchInterval: 3_000,
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

// --- Adapters ---
export function useAdapters() {
  return useQuery({
    queryKey: ["adapters"],
    queryFn: () => api<Array<Record<string, unknown>>>("/adapters"),
  });
}
