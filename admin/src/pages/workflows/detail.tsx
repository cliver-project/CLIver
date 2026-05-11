import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router";
import {
  useWorkflow,
  useSaveWorkflow,
  useRunWorkflow,
  useDeleteWorkflow,
  useAgents,
  useExecutions,
  useExecutionStatus,
  useResumeFromStep,
} from "@/hooks/use-api";
import { WorkflowCanvas, type LayoutData } from "./components/workflow-canvas";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useTranslation } from "@/i18n";

interface Step {
  id: string;
  type: "llm" | "python";
  prompt?: string;
  agent?: string;
  output_format?: string;
  file?: string;
  code?: string;
  depends_on?: string[];
  condition?: string;
  tools?: string[];
}

export default function WorkflowDetailPage() {
  const { t } = useTranslation();
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const { data, isLoading } = useWorkflow(name ?? "");
  const saveWorkflow = useSaveWorkflow(name ?? "");
  const runWorkflow = useRunWorkflow(name ?? "");
  const deleteWorkflow = useDeleteWorkflow(name ?? "");
  const { data: agentsData } = useAgents();
  const { data: executionsData } = useExecutions(name);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [saved, setSaved] = useState(false);
  const [executionId, setExecutionId] = useState<string | null>(null);

  const execStatus = useExecutionStatus(name ?? "", executionId ?? "");
  const execData = execStatus.data as Record<string, unknown> | undefined;
  const stepStatuses = execData?.step_statuses as Record<string, string> | undefined;
  const stepOutputs = execData?.step_outputs as Record<string, Record<string, unknown>> | undefined;
  const execOutputsDir = execData?.outputs_dir as string | undefined;
  const resumeFromStepMutation = useResumeFromStep(name ?? "");

  const [runningStepId, setRunningStepId] = useState<string | null>(null);
  const [streamStepId, setStreamStepId] = useState<string | null>(null);
  const [stepStreamOutput, setStepStreamOutput] = useState<string>("");
  const abortRef = useRef<AbortController | null>(null);

  const executions = useMemo(() => (executionsData ?? []) as Array<Record<string, unknown>>, [executionsData]);
  const isRunning = runWorkflow.isPending || execData?.status === "running";

  useEffect(() => {
    if (saveWorkflow.isSuccess) {
      setSaved(true);
      const timer = setTimeout(() => setSaved(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [saveWorkflow.isSuccess]);

  const handleRunStep = useCallback(async (stepId: string) => {
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setRunningStepId(stepId);
    setStreamStepId(stepId);
    setStepStreamOutput("");

    try {
      const res = await fetch(`/admin/api/workflows/${encodeURIComponent(name ?? "")}/steps/${encodeURIComponent(stepId)}/run`, {
        method: "POST",
        credentials: "include",
        signal: controller.signal,
      });
      if (!res.ok) {
        const text = await res.text();
        setStepStreamOutput(`Error: ${res.status} ${text}`);
        setRunningStepId(null);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setStepStreamOutput("Error: No response body");
        setRunningStepId(null);
        return;
      }

      const decoder = new TextDecoder();
      let accumulated = "";
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === "chunk" && evt.content) {
              accumulated += evt.content;
              setStepStreamOutput(accumulated);
            } else if (evt.type === "tool_start") {
              const toolLine = `\n[Tool: ${evt.tool}]\n`;
              accumulated += toolLine;
              setStepStreamOutput(accumulated);
            } else if (evt.type === "tool_error") {
              accumulated += `\n[Error: ${evt.error}]\n`;
              setStepStreamOutput(accumulated);
            } else if (evt.type === "error") {
              accumulated += `\n[Error: ${evt.message}]\n`;
              setStepStreamOutput(accumulated);
            } else if (evt.type === "done" && evt.content && !accumulated) {
              setStepStreamOutput(evt.content);
            }
          } catch { /* skip non-JSON lines */ }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setStepStreamOutput((prev) => prev + `\n[Error: ${(err as Error).message}]`);
      }
    } finally {
      setRunningStepId(null);
      abortRef.current = null;
    }
  }, [name]);

  if (isLoading) return <p className="text-muted-foreground p-6">{t("common.loading")}</p>;
  if (!data?.name) return <p className="text-muted-foreground p-6">{t("workflows.workflowNotFound")}</p>;

  const workflow = data as Record<string, unknown>;
  const steps = (workflow.steps ?? []) as Step[];
  const agentList = ((agentsData ?? []) as Array<Record<string, unknown>>).map((a) => String(a.name));
  const layout = (workflow.layout ?? null) as Record<string, unknown> | null;

  function handleSave(updatedSteps: Step[], updatedLayout: LayoutData) {
    saveWorkflow.mutate({ ...workflow, steps: updatedSteps, layout: updatedLayout });
  }

  function handleRun() {
    runWorkflow.mutate(undefined, {
      onSuccess: (result) => {
        const eid = (result as Record<string, unknown>)?.execution_id as string;
        if (eid) setExecutionId(eid);
      },
    });
  }

  function handleResumeFromStep(stepId: string) {
    if (!executionId) return;
    resumeFromStepMutation.mutate({ stepId, executionId });
  }

  return (
    <div className="-m-6">
      <WorkflowCanvas
        name={String(workflow.name ?? name)}
        steps={steps}
        agents={agentList}
        layout={layout}
        stepStatuses={stepStatuses}
        stepOutputs={stepOutputs}
        outputsDir={execOutputsDir ?? (workflow._default_outputs_dir as string | undefined)}
        onSave={handleSave}
        onRun={handleRun}
        onRunStep={handleRunStep}
        onResumeFromStep={handleResumeFromStep}
        runningStepId={runningStepId}
        streamStepId={streamStepId}
        stepStreamOutput={stepStreamOutput}
        onDelete={() => setConfirmDelete(true)}
        saving={saveWorkflow.isPending}
        saved={saved}
        running={isRunning}
        executions={executions}
        selectedExecutionId={executionId}
        onSelectExecution={setExecutionId}
      />
      <ConfirmDialog
        open={confirmDelete}
        title={t("workflows.deleteWorkflow")}
        description={t("workflows.deleteWorkflowDescription", { name: name ?? "" })}
        destructive
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => {
          setConfirmDelete(false);
          deleteWorkflow.mutate(undefined, {
            onSuccess: () => navigate("/admin/workflows"),
          });
        }}
      />
    </div>
  );
}
