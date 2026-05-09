import { useParams } from "react-router";
import { useWorkflow, useSaveWorkflow, useRunWorkflow } from "@/hooks/use-api";
import { WorkflowCanvas } from "./components/workflow-canvas";
import { useTranslation } from "@/i18n";

interface Step {
  id: string;
  type: "llm" | "python";
  prompt?: string;
  model?: string;
  role?: string;
  output_format?: string;
  file?: string;
  depends_on?: string[];
  condition?: string;
  tools?: string[];
}

export default function WorkflowDetailPage() {
  const { t } = useTranslation();
  const { name } = useParams<{ name: string }>();
  const { data, isLoading } = useWorkflow(name ?? "");
  const saveWorkflow = useSaveWorkflow(name ?? "");
  const runWorkflow = useRunWorkflow(name ?? "");

  if (isLoading) return <p className="text-muted-foreground p-6">{t("common.loading")}</p>;
  if (!data?.workflow) return <p className="text-muted-foreground p-6">{t("workflows.workflowNotFound")}</p>;

  const workflow = data.workflow as Record<string, unknown>;
  const steps = (workflow.steps ?? []) as Step[];
  const models = (data.models ?? []) as string[];

  function handleSave(updatedSteps: Step[]) {
    saveWorkflow.mutate({ ...workflow, steps: updatedSteps });
  }

  function handleRun() {
    runWorkflow.mutate(undefined);
  }

  return (
    <div className="-m-6">
      <WorkflowCanvas
        name={String(workflow.name ?? name)}
        steps={steps}
        models={models}
        onSave={handleSave}
        onRun={handleRun}
        saving={saveWorkflow.isPending}
      />
    </div>
  );
}
