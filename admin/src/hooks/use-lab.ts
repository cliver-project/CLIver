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

export interface Lab {
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

export interface LabSummary {
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

// --- Lab CRUD ---

export function useLabs() {
  return useQuery({
    queryKey: ["labs"],
    queryFn: () => api<LabSummary[]>("/labs"),
  });
}

export function useLab(id: string) {
  return useQuery({
    queryKey: ["lab", id],
    queryFn: () => api<Lab>(`/labs/${encodeURIComponent(id)}`),
    enabled: !!id,
  });
}

export function useCreateLab() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      title: string;
      description?: string;
      scenario_id?: string;
      default_agent?: string;
      context?: Record<string, unknown>;
      cells?: Partial<Cell>[];
    }) => apiPost<Lab>("/labs", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["labs"] });
    },
  });
}

export function useUpdateLab(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<Lab>) =>
      apiPut<Lab>(`/labs/${encodeURIComponent(id)}`, data),
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

// --- Cell Operations ---

export function useExecuteCell(labId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (cellId: string) =>
      apiPost<{ cell_id: string; status: string; outputs: Record<string, unknown> }>(
        `/labs/${encodeURIComponent(labId)}/cells/${encodeURIComponent(cellId)}/execute`,
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lab", labId] });
    },
  });
}

export function useRunAll(labId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiPost<{ status: string }>(`/labs/${encodeURIComponent(labId)}/run`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lab", labId] });
    },
  });
}

export function useAvailableRefs(labId: string, cellId: string) {
  return useQuery({
    queryKey: ["available-refs", labId, cellId],
    queryFn: () =>
      api<RefGroup[]>(
        `/labs/${encodeURIComponent(labId)}/cells/${encodeURIComponent(cellId)}/available-refs`,
      ),
    enabled: !!labId && !!cellId,
  });
}
