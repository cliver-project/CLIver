import { useState, useEffect, useCallback } from "react";
import { useParams, Link } from "react-router";
import { ArrowLeft, Save, Loader2, Lock, CheckCircle, XCircle, X, Eye, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { CodeEditor } from "@/components/code-editor";
import { MarkdownView } from "@/components/markdown-view";
import { useSkill, useSaveSkill } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import { type SkillForm, emptySkillForm, parseSkillMd, buildSkillMd } from "@/lib/skill-utils";

export default function SkillDetailPage() {
  const { t } = useTranslation();
  const { name } = useParams<{ name: string }>();
  const { data, isLoading } = useSkill(name ?? "");
  const saveSkill = useSaveSkill(name ?? "");

  const skill = data as Record<string, unknown> | undefined;
  const editable = skill?.editable === true;
  const rawContent = String(skill?.raw_content ?? "");

  const [form, setForm] = useState<SkillForm>(emptySkillForm());
  const [body, setBody] = useState("");
  const [dirty, setDirty] = useState(false);
  const [toolInput, setToolInput] = useState("");

  const [previewMode, setPreviewMode] = useState(!editable);

  useEffect(() => {
    if (rawContent) {
      const parsed = parseSkillMd(rawContent);
      setForm(parsed.form);
      setBody(parsed.body);

      setDirty(false);
    }
  }, [rawContent]);

  const markDirty = useCallback(() => {
    setDirty(true);
  }, []);

  const updateForm = useCallback(
    (field: keyof SkillForm, value: string | string[]) => {
      setForm((prev) => ({ ...prev, [field]: value }));
      markDirty();
    },
    [markDirty],
  );

  const handleBodyChange = useCallback(
    (value: string) => {
      setBody(value);
      markDirty();
    },
    [markDirty],
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
      updateForm(
        "allowedTools",
        form.allowedTools.filter((t) => t !== tool),
      );
    },
    [form.allowedTools, updateForm],
  );

  const handleSave = () => {
    const content = buildSkillMd(form, body);
    saveSkill.mutate(
      { content, path: String(skill?.path ?? "") },
      {
        onSuccess: () => {
          setDirty(false);
        },
      },
    );
  };

  if (isLoading)
    return <p className="text-muted-foreground">{t("common.loading")}</p>;
  if (!skill)
    return <p className="text-muted-foreground">{t("skills.skillNotFound")}</p>;

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
          <h1 className="text-2xl font-bold">{name}</h1>
          <div className="flex items-center gap-2 mt-1">
            {!!skill.source && (
              <Badge variant="secondary">{String(skill.source)}</Badge>
            )}
            {!editable && (
              <Badge variant="outline" className="gap-1">
                <Lock className="w-3 h-3" />
                {t("skills.readOnly")}
              </Badge>
            )}
          </div>
        </div>
        {editable && (
          <Button
            size="sm"
            onClick={handleSave}
            disabled={!dirty || saveSkill.isPending}
          >
            {saveSkill.isPending ? (
              <Loader2 className="w-4 h-4 mr-1 animate-spin" />
            ) : (
              <Save className="w-4 h-4 mr-1" />
            )}
            {saveSkill.isPending ? t("skills.saving") : t("skills.save")}
          </Button>
        )}
      </div>

      {/* Save feedback */}
      {saveSkill.isSuccess && !dirty && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-sm">
          <CheckCircle className="w-4 h-4 shrink-0" />
          {t("skills.saved")}
        </div>
      )}
      {saveSkill.isError && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-sm">
          <XCircle className="w-4 h-4 shrink-0" />
          {t("skills.saveError", {
            error: saveSkill.error?.message ?? "Unknown error",
          })}
        </div>
      )}

      {/* Metadata form */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t("skills.metadata")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Read-only info */}
          {!!skill.source && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground w-28 shrink-0">
                {t("skills.source")}
              </span>
              <Badge variant="secondary">{String(skill.source)}</Badge>
            </div>
          )}
          {!!skill.path && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground w-28 shrink-0">
                {t("skills.path")}
              </span>
              <span className="font-mono text-xs break-all">
                {String(skill.path)}
              </span>
            </div>
          )}

          {/* Editable fields */}
          <div className="space-y-1">
            <Label htmlFor="skill-name">{t("skills.name")}</Label>
            <Input
              id="skill-name"
              value={form.name}
              onChange={(e) => updateForm("name", e.target.value)}
              readOnly={!editable}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="skill-desc">{t("skills.description")}</Label>
            <Textarea
              id="skill-desc"
              value={form.description}
              onChange={(e) => updateForm("description", e.target.value)}
              readOnly={!editable}
              rows={2}
            />
          </div>
          <div className="space-y-1">
            <Label>{t("skills.allowedTools")}</Label>
            <div className="flex flex-wrap gap-1 mb-2">
              {form.allowedTools.map((tool) => (
                <Badge key={tool} variant="outline" className="gap-1">
                  {tool}
                  {editable && (
                    <button
                      onClick={() => removeTool(tool)}
                      className="ml-0.5 hover:text-destructive"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </Badge>
              ))}
              {form.allowedTools.length === 0 && (
                <span className="text-sm text-muted-foreground">
                  {t("skills.noToolsConfigured")}
                </span>
              )}
            </div>
            {editable && (
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
            )}
          </div>
        </CardContent>
      </Card>

      {/* Body editor / preview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center justify-between">
            {t("skills.bodyContent")}
            <Button variant="ghost" size="sm" onClick={() => setPreviewMode(!previewMode)}>
              {previewMode ? <Pencil className="w-3.5 h-3.5 mr-1" /> : <Eye className="w-3.5 h-3.5 mr-1" />}
              {previewMode ? t("common.edit") : t("skills.preview")}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {previewMode ? (
            <MarkdownView content={body} />
          ) : (
            <CodeEditor
              value={body}
              onChange={handleBodyChange}
              readOnly={!editable}
              language="markdown"
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
