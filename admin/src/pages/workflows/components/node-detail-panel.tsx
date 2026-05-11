import { useEffect, useRef, useState } from "react";
import { X, Play, SkipForward, Loader2 } from "lucide-react";
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

interface StepOutput {
  result?: string;
  files?: Array<{ type: string; name: string; path: string }>;
}

interface NodeDetailPanelProps {
  data: WorkflowNodeData;
  dependsOn: string[];
  agents: string[];
  stepOutput?: StepOutput;
  onChange: (updates: Partial<WorkflowNodeData>) => void;
  onRename: (oldId: string, newId: string) => void;
  onRunStep?: () => void;
  onResumeFromStep?: () => void;
  isRunning?: boolean;
  streamOutput?: string;
  onClose: () => void;
}

export function NodeDetailPanel({
  data,
  dependsOn,
  agents,
  stepOutput,
  onChange,
  onRename,
  onRunStep,
  onResumeFromStep,
  isRunning,
  streamOutput,
  onClose,
}: NodeDetailPanelProps) {
  const { t } = useTranslation();
  const isLlm = data.type === "llm";
  const [activeTab, setActiveTab] = useState<string>("edit");
  const outputRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (isRunning) setActiveTab("output");
  }, [isRunning]);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [streamOutput]);

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

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
        <TabsList className="mx-3 mt-2">
          <TabsTrigger value="edit" className="text-xs">{t("common.edit")}</TabsTrigger>
          <TabsTrigger value="output" className="text-xs">
            {t("common.output")}
            {isRunning && <Loader2 className="w-3 h-3 ml-1 animate-spin" />}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="edit" className="flex-1 overflow-y-auto px-3 pb-3 space-y-3">
          <div>
            <Label className="text-xs">{t("workflows.stepName")}</Label>
            <Input
              key={data.stepId}
              defaultValue={data.stepId}
              onBlur={(e) => {
                const v = e.target.value.trim().replace(/[^a-z0-9_]/g, "_");
                if (v && v !== data.stepId) onRename(data.stepId, v);
              }}
              onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
              className="h-8 text-xs font-mono"
            />
          </div>
          <div>
            <Label className="text-xs">{t("workflows.type")}</Label>
            <Input value={data.type} readOnly className="h-8 text-xs bg-muted" />
          </div>

          {isLlm && (
            <>
              <div>
                <Label className="text-xs">{t("tasks.taskAgent")}</Label>
                <Select
                  value={data.agent ?? "__default__"}
                  onValueChange={(v) => onChange({ agent: v === "__default__" ? undefined : v })}
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue placeholder={t("tasks.taskAgentDefault")} />
                  </SelectTrigger>
                  <SelectContent position="popper" className="z-[100] bg-card border shadow-lg">
                    <SelectItem value="__default__" className="text-xs">{t("tasks.taskAgentDefault")}</SelectItem>
                    {agents.map((a) => (
                      <SelectItem key={a} value={a} className="text-xs">{a}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
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
            <>
              <div>
                <Label className="text-xs">{t("workflows.file")}</Label>
                <Input
                  value={data.file ?? ""}
                  onChange={(e) => onChange({ file: e.target.value })}
                  className="h-8 text-xs font-mono"
                  placeholder="./scripts/run.py"
                  disabled={!!data.code}
                />
              </div>
              <div>
                <Label className="text-xs">{t("workflows.code")}</Label>
                <Textarea
                  value={data.code ?? ""}
                  onChange={(e) => onChange({ code: e.target.value })}
                  className="text-xs font-mono min-h-[140px]"
                  disabled={!!data.file}
                />
                <p className="text-[10px] text-muted-foreground mt-1">{t("workflows.codeHelp")}</p>
              </div>
            </>
          )}

          <div>
            <Label className="text-xs">{t("workflows.dependsOn")}</Label>
            <p className="text-xs text-muted-foreground mt-1">
              {dependsOn.length > 0 ? dependsOn.join(", ") : t("workflows.noneRootNode")}
            </p>
          </div>

          {(onRunStep || onResumeFromStep) && (
            <div className="flex gap-2 pt-2 border-t">
              {onRunStep && (
                <Button size="sm" variant="outline" className="text-xs flex-1" onClick={onRunStep} disabled={isRunning}>
                  {isRunning ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Play className="w-3 h-3 mr-1" />}
                  {isRunning ? "Running…" : "Run Step"}
                </Button>
              )}
              {onResumeFromStep && (
                <Button size="sm" variant="outline" className="text-xs flex-1" onClick={onResumeFromStep}>
                  <SkipForward className="w-3 h-3 mr-1" /> Resume
                </Button>
              )}
            </div>
          )}
        </TabsContent>

        <TabsContent value="output" className="flex-1 overflow-y-auto px-3 pb-3 space-y-3">
          {isRunning && streamOutput !== undefined ? (
            <div>
              <Label className="text-xs flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" />
                {t("workflows.running")}
              </Label>
              <pre
                ref={outputRef}
                className="text-xs font-mono bg-muted rounded p-2 mt-1 whitespace-pre-wrap max-h-[300px] overflow-y-auto"
              >
                {streamOutput || t("workflows.waitingForOutput")}
              </pre>
            </div>
          ) : streamOutput ? (
            <div>
              <Label className="text-xs">{t("workflows.textResult")}</Label>
              <pre className="text-xs font-mono bg-muted rounded p-2 mt-1 whitespace-pre-wrap max-h-[300px] overflow-y-auto">
                {streamOutput}
              </pre>
            </div>
          ) : !stepOutput ? (
            <p className="text-xs text-muted-foreground">
              {t("workflows.runToSeeOutput")}
            </p>
          ) : (
            <>
              {stepOutput.result && (
                <div>
                  <Label className="text-xs">{t("workflows.textResult")}</Label>
                  <pre className="text-xs font-mono bg-muted rounded p-2 mt-1 whitespace-pre-wrap max-h-[200px] overflow-y-auto">
                    {stepOutput.result}
                  </pre>
                </div>
              )}
              {stepOutput.files && stepOutput.files.length > 0 && (
                <div>
                  <Label className="text-xs">{t("workflows.outputFiles")}</Label>
                  <div className="space-y-1 mt-1">
                    {stepOutput.files.map((f, i) => (
                      <div key={i} className="text-xs">
                        {f.type === "image" ? (
                          <div>
                            <img
                              src={`/admin/api/media/${encodeURIComponent(f.path)}`}
                              alt={f.name}
                              className="rounded max-w-full max-h-[150px] mt-1"
                            />
                            <span className="text-muted-foreground">{f.name}</span>
                          </div>
                        ) : f.type === "audio" ? (
                          <div>
                            <audio controls src={`/admin/api/media/${encodeURIComponent(f.path)}`} className="w-full h-8 mt-1" />
                            <span className="text-muted-foreground">{f.name}</span>
                          </div>
                        ) : f.type === "video" ? (
                          <div>
                            <video controls src={`/admin/api/media/${encodeURIComponent(f.path)}`} className="rounded max-w-full max-h-[150px] mt-1" />
                            <span className="text-muted-foreground">{f.name}</span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground font-mono">{f.name}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {!stepOutput.result && (!stepOutput.files || stepOutput.files.length === 0) && (
                <p className="text-xs text-muted-foreground">{t("workflows.noOutput")}</p>
              )}
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
