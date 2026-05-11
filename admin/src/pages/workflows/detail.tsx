import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router";
import {
  useWorkflow,
  useSaveWorkflow,
  useRunWorkflow,
  useDeleteWorkflow,
  useAgents,
  useExecutions,
  useExecutionStatus,
  useRunStep,
  useResumeFromStep,
} from "@/hooks/use-api";
import { WorkflowCanvas } from "./components/workflow-canvas";
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
  const runStepMutation = useRunStep(name ?? "", "");
  const resumeFromStepMutation = useResumeFromStep(name ?? "");

  const executions = useMemo(() => (executionsData ?? []) as Array<Record<string, unknown>>, [executionsData]);
  const isRunning = runWorkflow.isPending || execData?.status === "running";

  useEffect(() => {
    if (saveWorkflow.isSuccess) {
      setSaved(true);
      const timer = setTimeout(() => setSaved(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [saveWorkflow.isSuccess]);

  if (isLoading) return <p className="text-muted-foreground p-6">{t("common.loading")}</p>;
  if (!data?.name) return <p className="text-muted-foreground p-6">{t("workflows.workflowNotFound")}</p>;

  const workflow = data as Record<string, unknown>;
  const steps = (workflow.steps ?? []) as Step[];
  const agentList = ((agentsData ?? []) as Array<Record<string, unknown>>).map((a) => String(a.name));
  const layout = (workflow.layout ?? null) as Record<string, unknown> | null;

  function handleSave(updatedSteps: Step[], updatedLayout: Record<string, unknown>) {
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

  function handleRunStep(stepId: string) {
    runStepMutation.mutate();
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
