import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, apiPost, apiPut, apiDelete } from "@/lib/api";

// --- Types ---

export interface Cell {
  id: string;
  type: "config" | "llm" | "code" | "display";
  title: string;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  status: "idle" | "running" | "completed" | "error";
  error?: string | null;
  duration_ms: number;
}

export interface Notebook {
  $schema: string;
  id: string;
  title: string;
  description: string;
  scenario_id?: string | null;
  default_agent?: string | null;
  context: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  cells: Cell[];
}

export interface NotebookSummary {
  id: string;
  title: string;
  description: string;
  scenario_id?: string | null;
  cell_count: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface RefField {
  path: string;
  preview: string;
  type: string;
}

export interface RefGroup {
  cell_id: string;
  cell_title: string;
  fields: RefField[];
}

// --- Notebook CRUD ---

export function useNotebooks() {
  return useQuery({
    queryKey: ["notebooks"],
    queryFn: () => api<NotebookSummary[]>("/notebooks"),
  });
}

export function useNotebook(id: string) {
  return useQuery({
    queryKey: ["notebook", id],
    queryFn: () => api<Notebook>(`/notebooks/${encodeURIComponent(id)}`),
    enabled: !!id,
  });
}

export function useCreateNotebook() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      title: string;
      description?: string;
      scenario_id?: string;
      default_agent?: string;
      context?: Record<string, unknown>;
      cells?: Partial<Cell>[];
    }) => apiPost<Notebook>("/notebooks", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notebooks"] });
    },
  });
}

export function useUpdateNotebook(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<Notebook>) =>
      apiPut<Notebook>(`/notebooks/${encodeURIComponent(id)}`, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notebook", id] });
      qc.invalidateQueries({ queryKey: ["notebooks"] });
    },
  });
}

export function useDeleteNotebook() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/notebooks/${encodeURIComponent(id)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notebooks"] });
    },
  });
}

// --- Cell Operations ---

export function useExecuteCell(notebookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (cellId: string) =>
      apiPost<{ cell_id: string; status: string; outputs: Record<string, unknown> }>(
        `/notebooks/${encodeURIComponent(notebookId)}/cells/${encodeURIComponent(cellId)}/execute`,
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notebook", notebookId] });
    },
  });
}

export function useRunAll(notebookId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiPost<{ status: string }>(`/notebooks/${encodeURIComponent(notebookId)}/run`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notebook", notebookId] });
    },
  });
}

export function useAvailableRefs(notebookId: string, cellId: string) {
  return useQuery({
    queryKey: ["available-refs", notebookId, cellId],
    queryFn: () =>
      api<RefGroup[]>(
        `/notebooks/${encodeURIComponent(notebookId)}/cells/${encodeURIComponent(cellId)}/available-refs`,
      ),
    enabled: !!notebookId && !!cellId,
  });
}
