import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useCreateWorkflow } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

export default function WorkflowCreatePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const createWorkflow = useCreateWorkflow();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createWorkflow.mutate(
      { name: name.trim(), description: description.trim() || undefined },
      {
        onSuccess: () => {
          navigate(`/admin/workflows/${encodeURIComponent(name.trim())}`);
        },
      },
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link to="/admin/workflows">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-4 h-4" />
          </Button>
        </Link>
        <h1 className="text-2xl font-bold">{t("workflows.createWorkflow")}</h1>
      </div>

      <form onSubmit={handleSubmit}>
        <Card>
          <CardContent className="pt-6 space-y-4">
            <div className="space-y-1">
              <Label>{t("workflows.workflowName")}</Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t("workflows.workflowNamePlaceholder")}
                required
                className="font-mono"
              />
            </div>

            <div className="space-y-1">
              <Label>{t("workflows.description")}</Label>
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder={t("workflows.descriptionPlaceholder")}
              />
            </div>

            {createWorkflow.isError && (
              <p className="text-sm text-red-500">
                {t("workflows.createError", { error: createWorkflow.error?.message ?? "Unknown" })}
              </p>
            )}

            <div className="flex justify-end">
              <Button type="submit" disabled={createWorkflow.isPending || !name.trim()}>
                {createWorkflow.isPending ? t("workflows.saving") : t("workflows.createWorkflow")}
              </Button>
            </div>
          </CardContent>
        </Card>
      </form>
    </div>
  );
}
