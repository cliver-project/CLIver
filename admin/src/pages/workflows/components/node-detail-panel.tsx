import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useTranslation } from "@/i18n";
import type { WorkflowNodeData } from "./workflow-node";

interface NodeDetailPanelProps {
  data: WorkflowNodeData;
  dependsOn: string[];
  models: string[];
  onChange: (updates: Partial<WorkflowNodeData>) => void;
  onClose: () => void;
}

export function NodeDetailPanel({
  data,
  dependsOn,
  models,
  onChange,
  onClose,
}: NodeDetailPanelProps) {
  const { t } = useTranslation();
  const isLlm = data.type === "llm";

  return (
    <div className="w-72 bg-card border-l border-border flex flex-col h-full overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-border">
        <div
          className="w-2 h-2 rounded-full"
          style={{ background: isLlm ? "#818cf8" : "#34d399" }}
        />
        <span className="font-semibold text-sm flex-1">{data.stepId}</span>
        <Button variant="ghost" size="icon" className="w-6 h-6" onClick={onClose}>
          <X className="w-3.5 h-3.5" />
        </Button>
      </div>

      <Tabs defaultValue="edit" className="flex-1 flex flex-col overflow-hidden">
        <TabsList className="mx-3 mt-2">
          <TabsTrigger value="edit" className="text-xs">{t("common.edit")}</TabsTrigger>
          <TabsTrigger value="output" className="text-xs">{t("common.output")}</TabsTrigger>
        </TabsList>

        <TabsContent value="edit" className="flex-1 overflow-y-auto px-3 pb-3 space-y-3">
          <div>
            <Label className="text-xs">{t("workflows.type")}</Label>
            <Input value={data.type} readOnly className="h-8 text-xs bg-muted" />
          </div>

          {isLlm && (
            <>
              <div>
                <Label className="text-xs">{t("workflows.model")}</Label>
                <Select
                  value={data.model ?? ""}
                  onValueChange={(v) => onChange({ model: v })}
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue placeholder="Default" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label className="text-xs">{t("workflows.role")}</Label>
                <Input
                  value={data.role ?? ""}
                  onChange={(e) => onChange({ role: e.target.value })}
                  className="h-8 text-xs"
                  placeholder="Optional role description"
                />
              </div>

              <div>
                <Label className="text-xs">{t("workflows.outputFormat")}</Label>
                <div className="flex gap-1">
                  {(["json", "text", "markdown"] as const).map((fmt) => (
                    <Button
                      key={fmt}
                      size="sm"
                      variant={data.outputFormat === fmt ? "default" : "outline"}
                      className="h-6 text-[10px] px-2"
                      onClick={() => onChange({ outputFormat: fmt })}
                    >
                      {fmt}
                    </Button>
                  ))}
                </div>
              </div>

              <div>
                <Label className="text-xs">{t("workflows.prompt")}</Label>
                <Textarea
                  value={data.prompt ?? ""}
                  onChange={(e) => onChange({ prompt: e.target.value })}
                  className="text-xs font-mono min-h-[120px]"
                />
              </div>
            </>
          )}

          {!isLlm && (
            <div>
              <Label className="text-xs">{t("workflows.file")}</Label>
              <Input
                value={data.file ?? ""}
                onChange={(e) => onChange({ file: e.target.value })}
                className="h-8 text-xs font-mono"
                placeholder="./scripts/run.py"
              />
            </div>
          )}

          <div>
            <Label className="text-xs">{t("workflows.dependsOn")}</Label>
            <p className="text-xs text-muted-foreground mt-1">
              {dependsOn.length > 0 ? dependsOn.join(", ") : t("workflows.noneRootNode")}
            </p>
          </div>
        </TabsContent>

        <TabsContent value="output" className="flex-1 overflow-y-auto px-3 pb-3">
          <p className="text-xs text-muted-foreground">
            {t("workflows.runToSeeOutput")}
          </p>
        </TabsContent>
      </Tabs>
    </div>
  );
}
