import { useState, useCallback, useRef } from "react";
import { Link, useNavigate } from "react-router";
import { ArrowLeft, Save, Loader2, XCircle, X, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { CodeEditor } from "@/components/code-editor";
import { MarkdownView } from "@/components/markdown-view";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { useCreateSkill } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import {
  type SkillForm,
  emptySkillForm,
  buildSkillMd,
  validateSkillName,
} from "@/lib/skill-utils";

export default function SkillCreatePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const createSkill = useCreateSkill();

  const [form, setForm] = useState<SkillForm>(emptySkillForm());
  const [body, setBody] = useState("\n");
  const [toolInput, setToolInput] = useState("");
  const [nameError, setNameError] = useState<string | null>(null);

  const [aiDialogOpen, setAiDialogOpen] = useState(false);
  const [aiPrompt, setAiPrompt] = useState("");
  const [aiGenerating, setAiGenerating] = useState(false);
  const [aiPreview, setAiPreview] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const generateWithAI = useCallback(async () => {
    if (!aiPrompt.trim()) return;
    setAiGenerating(true);
    setAiPreview("");
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/admin/api/chat", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: aiPrompt,
          system_message:
            "You are a skill content generator for CLIver, an AI CLI agent. " +
            "Generate ONLY the markdown body content for a SKILL.md file. " +
            "Do NOT include YAML frontmatter (name, description, allowed_tools) — those are handled separately. " +
            "Output ONLY the markdown content that goes after the frontmatter. " +
            "The content should be instructions for the AI agent on how to perform the skill. " +
            "Use clear headings, step-by-step processes, and guidelines. " +
            "Do NOT wrap the output in code fences.",
          filter_tools: [],
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

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
          const payload = line.slice(6);
          if (payload === "[DONE]") continue;
          try {
            const evt = JSON.parse(payload);
            if (evt.type === "chunk" && evt.content) {
              accumulated += evt.content;
              setAiPreview(accumulated);
            } else if (evt.type === "done" && evt.content && !accumulated) {
              setAiPreview(evt.content);
            }
          } catch { /* skip non-JSON lines */ }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setAiPreview((prev) => prev + `\n\n[Error: ${(err as Error).message}]`);
      }
    } finally {
      setAiGenerating(false);
      abortRef.current = null;
    }
  }, [aiPrompt]);

  const acceptAiContent = useCallback(() => {
    setBody(aiPreview);
    setAiDialogOpen(false);
    setAiPrompt("");
    setAiPreview("");
  }, [aiPreview]);

  const updateForm = useCallback(
    (field: keyof SkillForm, value: string | string[]) => {
      setForm((prev) => ({ ...prev, [field]: value }));
      if (field === "name") setNameError(null);
    },
    [],
  );

  const addTool = useCallback(() => {
    const tool = toolInput.trim();
    if (tool && !form.allowedTools.includes(tool)) {
      updateForm("allowedTools", [...form.allowedTools, tool]);
      setToolInput("");
    }
  }, [toolInput, form.allowedTools, updateForm]);

  const removeTool = useCallback(
    (tool: string) => {
      updateForm("allowedTools", form.allowedTools.filter((t) => t !== tool));
    },
    [form.allowedTools, updateForm],
  );

  const handleCreate = () => {
    const err = validateSkillName(form.name);
    if (err) {
      setNameError(err);
      return;
    }
    if (!form.description.trim()) {
      setNameError(t("skills.descriptionRequired"));
      return;
    }
    const content = buildSkillMd(form, body);
    createSkill.mutate(
      { name: form.name, content },
      {
        onSuccess: () => {
          navigate(`/admin/skills/${encodeURIComponent(form.name)}`);
        },
      },
    );
  };

  const canCreate = form.name.trim().length > 0 && form.description.trim().length > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to="/admin/skills">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-4 h-4" />
          </Button>
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold">{t("skills.createSkill")}</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {t("skills.createDescription")}
          </p>
        </div>
        <Button
          size="sm"
          onClick={handleCreate}
          disabled={!canCreate || createSkill.isPending}
        >
          {createSkill.isPending ? (
            <Loader2 className="w-4 h-4 mr-1 animate-spin" />
          ) : (
            <Save className="w-4 h-4 mr-1" />
          )}
          {createSkill.isPending ? t("skills.creating") : t("skills.create")}
        </Button>
      </div>

      {/* Error feedback */}
      {createSkill.isError && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-sm">
          <XCircle className="w-4 h-4 shrink-0" />
          {t("skills.createError", {
            error: createSkill.error?.message ?? "Unknown error",
          })}
        </div>
      )}

      {/* Metadata form */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t("skills.metadata")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <Label htmlFor="skill-name">{t("skills.name")} *</Label>
            <Input
              id="skill-name"
              value={form.name}
              onChange={(e) => updateForm("name", e.target.value.toLowerCase())}
              placeholder={t("skills.namePlaceholder")}
              className={nameError ? "border-destructive" : ""}
            />
            {nameError && (
              <p className="text-xs text-destructive">{nameError}</p>
            )}
            <p className="text-xs text-muted-foreground">
              {t("skills.nameHint")}
            </p>
          </div>
          <div className="space-y-1">
            <Label htmlFor="skill-desc">{t("skills.description")} *</Label>
            <Textarea
              id="skill-desc"
              value={form.description}
              onChange={(e) => updateForm("description", e.target.value)}
              placeholder={t("skills.descriptionPlaceholder")}
              rows={2}
            />
          </div>
          <div className="space-y-1">
            <Label>{t("skills.allowedTools")}</Label>
            <div className="flex flex-wrap gap-1 mb-2">
              {form.allowedTools.map((tool) => (
                <Badge key={tool} variant="outline" className="gap-1">
                  {tool}
                  <button
                    onClick={() => removeTool(tool)}
                    className="ml-0.5 hover:text-destructive"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </Badge>
              ))}
              {form.allowedTools.length === 0 && (
                <span className="text-sm text-muted-foreground">
                  {t("skills.noToolsConfigured")}
                </span>
              )}
            </div>
            <div className="flex gap-2">
              <Input
                value={toolInput}
                onChange={(e) => setToolInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addTool();
                  }
                }}
                placeholder={t("skills.addToolPlaceholder")}
                className="flex-1"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={addTool}
                disabled={!toolInput.trim()}
              >
                {t("skills.addTool")}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Body editor */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center justify-between">
            {t("skills.bodyContent")}
            <Button variant="outline" size="sm" onClick={() => setAiDialogOpen(true)}>
              <Sparkles className="w-3.5 h-3.5 mr-1" />
              {t("skills.generateWithAI")}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <CodeEditor
            value={body}
            onChange={setBody}
            language="markdown"
          />
        </CardContent>
      </Card>

      {/* AI Generation Dialog */}
      <Dialog open={aiDialogOpen} onOpenChange={(open) => {
        if (!open && aiGenerating) {
          abortRef.current?.abort();
        }
        if (!open) {
          setAiDialogOpen(false);
          setAiGenerating(false);
        }
      }}>
        <DialogContent className="max-w-2xl max-h-[80vh] flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              {t("skills.generateWithAI")}
            </DialogTitle>
            <DialogDescription>
              {t("skills.aiGenerateDesc")}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 flex-1 overflow-hidden flex flex-col">
            <div className="flex gap-2">
              <Textarea
                value={aiPrompt}
                onChange={(e) => setAiPrompt(e.target.value)}
                placeholder={t("skills.aiPromptPlaceholder")}
                rows={3}
                disabled={aiGenerating}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    generateWithAI();
                  }
                }}
              />
            </div>
            <Button
              size="sm"
              onClick={generateWithAI}
              disabled={aiGenerating || !aiPrompt.trim()}
              className="self-end"
            >
              {aiGenerating ? (
                <Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />
              ) : (
                <Sparkles className="w-3.5 h-3.5 mr-1" />
              )}
              {aiGenerating ? t("skills.aiGenerating") : t("skills.aiGenerate")}
            </Button>
            {aiPreview && (
              <div className="flex-1 overflow-auto min-h-[200px] max-h-[400px] rounded-md border border-input bg-muted/30 p-3">
                <MarkdownView content={aiPreview} />
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { abortRef.current?.abort(); setAiDialogOpen(false); }}>
              {t("common.cancel")}
            </Button>
            <Button onClick={acceptAiContent} disabled={!aiPreview || aiGenerating}>
              {t("skills.aiAccept")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
