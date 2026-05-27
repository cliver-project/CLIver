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
export function useSessions() {
  return useQuery({
    queryKey: ["sessions"],
    queryFn: () => api<Array<Record<string, unknown>>>("/sessions"),
  });
}

export function useSessionTurns(sessionId: string) {
  return useQuery({
    queryKey: ["session-turns", sessionId],
    queryFn: () =>
      api<Array<{ role: string; content: string; timestamp?: string }>>(
        `/sessions/${encodeURIComponent(sessionId)}`,
      ),
    enabled: !!sessionId,
  });
}

export function useDeleteSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) =>
      apiDelete(`/sessions/${encodeURIComponent(sessionId)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
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

// --- Agents (DB-backed) ---

export interface AgentInfo {
  id: string;
  name: string;
  type: string;
  description?: string | null;
  role?: string | null;
  model?: string | null;
  is_default: number;
  created_at: string;
  updated_at: string;
}

export function useAgents() {
  return useQuery({
    queryKey: ["agents"],
    queryFn: () => api<AgentInfo[]>("/agents"),
  });
}

export function useAgent(id: string | undefined) {
  return useQuery({
    queryKey: ["agent", id],
    queryFn: () => api<AgentInfo>(`/agents/${encodeURIComponent(id!)}`),
    enabled: !!id,
  });
}

export function useCreateAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<AgentInfo>) =>
      apiPost<AgentInfo>("/agents", data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agents"] }),
  });
}

export function useUpdateAgent(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<AgentInfo>) =>
      api(`/agents/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent", id] });
      qc.invalidateQueries({ queryKey: ["agents"] });
    },
  });
}

export function useDeleteAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/agents/${encodeURIComponent(id)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agents"] }),
  });
}

export function useSetDefaultAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      apiPost<{ status: string }>(`/agents/${encodeURIComponent(id)}/default`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agents"] }),
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

// --- Models, Providers, Endpoints ---

export interface ModelProvider {
  id: string;
  name: string;
  type: string;
  api_key?: string;
  api_url?: string;
  rate_limit?: { requests: number; period: string; margin: number };
  pricing?: { currency?: string; input?: number; output?: number; cached_input?: number };
  created_at: string;
  updated_at: string;
}

export interface ModelInfo {
  id: string;
  provider: string;
  name: string;
  category: string;
  model: string;
  api_url: string | null;
  options: Record<string, unknown>;
  is_default: number;
}

export function useModels(category?: string) {
  const params = category ? `?category=${encodeURIComponent(category)}` : "";
  return useQuery({
    queryKey: ["models", category],
    queryFn: () => api<ModelInfo[]>(`/models${params}`),
  });
}

export function useCreateModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<ModelInfo>) =>
      apiPost<ModelInfo>("/models", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useUpdateModel(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<ModelInfo>) =>
      api(`/models/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useDeleteModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/models/${encodeURIComponent(id)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
  });
}

export function useSetDefaultModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiPost<{ status: string }>(`/models/${encodeURIComponent(id)}/default`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
  });
}

export function useProviders() {
  return useQuery({
    queryKey: ["providers"],
    queryFn: () => api<ModelProvider[]>("/providers"),
  });
}

export function useCreateProvider() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<ModelProvider>) =>
      apiPost<ModelProvider>("/providers", data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["providers"] }),
  });
}

export function useUpdateProvider(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<ModelProvider>) =>
      api(`/providers/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["providers"] });
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useDeleteProvider() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/providers/${encodeURIComponent(id)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["providers"] });
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

