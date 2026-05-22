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
      qc.invalidateQueries({ queryKey: ["agents"] });
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

// --- AI Labs ---

export interface Lab {
  id: string;
  title: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface GoldenTest {
  id: string;
  lab_id: string;
  name: string;
  input: string;
  expected_output: string;
  expected_files: string;
  sort_order: number;
}

export interface LabDetail {
  lab: Lab;
  sessions: Array<Record<string, unknown>>;
  session_id: string | null;
}

export interface TestRunResult {
  test_id: string;
  name: string;
  input: string;
  expected_output: string;
  actual_output: string;
  expected_files: string;
}

export function useLabs() {
  return useQuery({
    queryKey: ["labs"],
    queryFn: () => api<Lab[]>("/labs"),
  });
}

export function useLab(id: string | undefined) {
  return useQuery({
    queryKey: ["lab", id],
    queryFn: () => api<LabDetail>(`/labs/${encodeURIComponent(id!)}`),
    enabled: !!id,
  });
}

export function useCreateLab() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { title: string; description?: string }) =>
      apiPost<Lab>("/labs", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["labs"] });
    },
  });
}

export function useUpdateLab(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { title?: string; description?: string }) =>
      api(`/labs/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lab", id] });
      qc.invalidateQueries({ queryKey: ["labs"] });
    },
  });
}

export function useDeleteLab() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/labs/${encodeURIComponent(id)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["labs"] });
    },
  });
}

export function useLabGoldenTests(labId: string | undefined) {
  return useQuery({
    queryKey: ["lab-golden-tests", labId],
    queryFn: () => api<GoldenTest[]>(`/labs/${encodeURIComponent(labId!)}/golden-tests`),
    enabled: !!labId,
  });
}

export function useCreateGoldenTest(labId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; input: string; expected_output: string; expected_files?: string }) =>
      apiPost<GoldenTest>(`/labs/${encodeURIComponent(labId)}/golden-tests`, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lab-golden-tests", labId] });
    },
  });
}

export function useDeleteGoldenTest(labId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (testId: string) =>
      apiDelete(`/labs/${encodeURIComponent(labId)}/golden-tests/${encodeURIComponent(testId)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lab-golden-tests", labId] });
    },
  });
}

export function useRunGoldenTests(labId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiPost<{ results: TestRunResult[] }>(`/labs/${encodeURIComponent(labId)}/golden-tests/run`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lab-golden-tests", labId] });
    },
  });
}

// --- MCP Servers ---

export interface MCPServer {
  id: string;
  name: string;
  transport: string;
  url?: string;
  auth?: string;
  headers?: string;
  command?: string;
  args?: string;
  envs?: string;
  created_at: string;
  updated_at: string;
}

export function useMCPServers() {
  return useQuery({
    queryKey: ["mcp-servers"],
    queryFn: () => api<MCPServer[]>("/mcp-servers"),
  });
}

export function useMCPServer(id: string | undefined) {
  return useQuery({
    queryKey: ["mcp-server", id],
    queryFn: () => api<MCPServer>(`/mcp-servers/${encodeURIComponent(id!)}`),
    enabled: !!id,
  });
}

export function useCreateMCPServer() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<MCPServer>) =>
      apiPost<MCPServer>("/mcp-servers", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["mcp-servers"] });
    },
  });
}

export function useUpdateMCPServer(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<MCPServer>) =>
      api(`/mcp-servers/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["mcp-server", id] });
      qc.invalidateQueries({ queryKey: ["mcp-servers"] });
    },
  });
}

export function useDeleteMCPServer() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/mcp-servers/${encodeURIComponent(id)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["mcp-servers"] });
    },
  });
}
