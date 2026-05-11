import { useState, useEffect, useCallback, useRef } from "react";
import {
  Save, Loader2, CheckCircle, XCircle, Plus, Trash2, X, ChevronDown, ChevronRight,
  Zap, AlertTriangle, RotateCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useConfig, useModels, useSaveConfig, useTestProvider, useAdapters, useRestartGateway } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

type Config = Record<string, unknown>;
type Provider = {
  type: string; api_url: string; api_key: string;
  rate_limit?: { requests: number; period: string; margin: number };
  pricing?: { currency?: string; input?: number; output?: number; cached_input?: number };
  image_url?: string; image_model?: string; audio_url?: string; audio_model?: string;
  models: Array<{
    name: string; options?: Record<string, number>; think_mode?: boolean | null;
    context_window?: number | null;
    pricing?: { currency?: string; input?: number; output?: number; cached_input?: number };
    capabilities?: string[];
  }>;
};
type MCPServer = {
  transport: string; command?: string; args?: string[]; env?: Record<string, string>;
  url?: string; headers?: Record<string, string>;
};
type PlatformAdapter = {
  type: string; token: string; app_token: string;
  home_channel?: string; allowed_users?: string[];
};
type Gateway = {
  host: string; port: number; admin_username: string; admin_password: string;
  log_file: string; log_max_bytes: number; log_backup_count: number;
  api_key?: string;
  platforms: Record<string, PlatformAdapter>;
};
type Session = { max_sessions: number; max_turns_per_session: number; max_age_days: number };

function TagInput({ tags, onChange, placeholder, addLabel, disabled }: {
  tags: string[]; onChange: (t: string[]) => void;
  placeholder: string; addLabel: string; disabled?: boolean;
}) {
  const [input, setInput] = useState("");
  const add = () => {
    const v = input.trim();
    if (v && !tags.includes(v)) { onChange([...tags, v]); setInput(""); }
  };
  return (
    <div>
      <div className="flex flex-wrap gap-1 mb-2">
        {tags.map((t) => (
          <Badge key={t} variant="outline" className="gap-1">
            {t}
            {!disabled && (
              <button onClick={() => onChange(tags.filter((x) => x !== t))} className="ml-0.5 hover:text-destructive">
                <X className="w-3 h-3" />
              </button>
            )}
          </Badge>
        ))}
      </div>
      {!disabled && (
        <div className="flex gap-2">
          <Input value={input} onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); add(); } }}
            placeholder={placeholder} className="flex-1" />
          <Button type="button" variant="outline" size="sm" onClick={add} disabled={!input.trim()}>
            {addLabel}
          </Button>
        </div>
      )}
    </div>
  );
}

function KVEditor({ entries, onChange, keyLabel, valueLabel, addLabel }: {
  entries: Record<string, string>; onChange: (e: Record<string, string>) => void;
  keyLabel: string; valueLabel: string; addLabel: string;
}) {
  const [newKey, setNewKey] = useState("");
  const [newVal, setNewVal] = useState("");
  const add = () => {
    if (newKey.trim()) {
      onChange({ ...entries, [newKey.trim()]: newVal });
      setNewKey(""); setNewVal("");
    }
  };
  return (
    <div className="space-y-2">
      {Object.entries(entries).map(([k, v]) => (
        <div key={k} className="flex gap-2 items-center">
          <Input value={k} readOnly className="w-1/3 text-xs" />
          <Input value={v} onChange={(e) => onChange({ ...entries, [k]: e.target.value })} className="flex-1 text-xs" />
          <Button variant="ghost" size="icon" className="shrink-0 h-8 w-8"
            onClick={() => { const copy = { ...entries }; delete copy[k]; onChange(copy); }}>
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
      ))}
      <div className="flex gap-2 items-center">
        <Input value={newKey} onChange={(e) => setNewKey(e.target.value)} placeholder={keyLabel} className="w-1/3 text-xs" />
        <Input value={newVal} onChange={(e) => setNewVal(e.target.value)} placeholder={valueLabel}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); add(); } }} className="flex-1 text-xs" />
        <Button variant="outline" size="sm" onClick={add} disabled={!newKey.trim()} className="shrink-0">
          {addLabel}
        </Button>
      </div>
    </div>
  );
}

function CollapsibleCard({ title, defaultOpen, onDelete, className, children }: {
  title: string; defaultOpen?: boolean; onDelete?: () => void; className?: string; children: React.ReactNode;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(defaultOpen ?? false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  return (
    <Card className={className}>
      <CardHeader className="pb-2 cursor-pointer" onClick={() => setOpen(!open)}>
        <CardTitle className="text-sm flex items-center gap-2">
          {open ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <span className="flex-1">{title}</span>
          {onDelete && (
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={(e) => { e.stopPropagation(); setConfirmOpen(true); }}>
              <Trash2 className="w-3 h-3 text-destructive" />
            </Button>
          )}
        </CardTitle>
      </CardHeader>
      {open && <CardContent className="space-y-3 [&>*:nth-child(even)]:bg-muted/20 [&>*:nth-child(even)]:rounded-md [&>*:nth-child(even)]:px-2 [&>*:nth-child(even)]:py-1">{children}</CardContent>}
      {onDelete && (
        <ConfirmDialog
          open={confirmOpen}
          title={t("config.deleteConfirmTitle", { name: title })}
          description={t("config.deleteConfirmDesc")}
          destructive
          onCancel={() => setConfirmOpen(false)}
          onConfirm={() => { setConfirmOpen(false); onDelete(); }}
        />
      )}
    </Card>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <Label className="text-xs">{label}</Label>
      {children}
    </div>
  );
}

function NumberInput({ value, onChange, ...props }: {
  value: number | undefined | null; onChange: (v: number | undefined) => void;
} & Omit<React.ComponentProps<typeof Input>, "value" | "onChange">) {
  return (
    <Input type="number" value={value ?? ""} onChange={(e) => {
      const v = e.target.value; onChange(v === "" ? undefined : Number(v));
    }} {...props} />
  );
}

// ─────────────────────────────────────────────
// Tab: General
// ─────────────────────────────────────────────

function GeneralTab({ config, setConfig, modelList }: {
  config: Config; setConfig: (c: Config) => void; modelList: string[];
}) {
  const { t } = useTranslation();
  const set = (k: string, v: unknown) => setConfig({ ...config, [k]: v });

  return (
    <div className="space-y-4">
      <Field label={t("config.defaultModel")}>
        <select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={String(config.default_model ?? "")} onChange={(e) => set("default_model", e.target.value)}>
          <option value="">—</option>
          {modelList.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
      </Field>
      <Field label={t("config.userAgent")}>
        <Input value={String(config.user_agent ?? "")} onChange={(e) => set("user_agent", e.target.value)} />
      </Field>
      <Field label={t("config.timezone")}>
        <Input value={String(config.timezone ?? "")} onChange={(e) => set("timezone", e.target.value)}
          placeholder={t("config.timezonePlaceholder")} />
      </Field>
      <Field label={t("config.searchEngines")}>
        <div className="flex flex-wrap gap-3">
          {["duckduckgo", "bing", "sogou", "google", "baidu"].map((engine) => {
            const engines = (config.search_engines as string[]) ?? [];
            const checked = engines.includes(engine);
            return (
              <div key={engine} className="flex items-center gap-1.5">
                <input type="checkbox" id={`engine-${engine}`} checked={checked}
                  onChange={() => set("search_engines", checked ? engines.filter((e) => e !== engine) : [...engines, engine])}
                  className="h-4 w-4 rounded border-input" />
                <Label htmlFor={`engine-${engine}`} className="text-sm">{engine}</Label>
              </div>
            );
          })}
        </div>
      </Field>
      <div className="flex items-center gap-3">
        <input type="checkbox" id="skill-auto" checked={Boolean(config.skill_auto_learn)}
          onChange={(e) => set("skill_auto_learn", e.target.checked)}
          className="h-4 w-4 rounded border-input" />
        <div>
          <Label htmlFor="skill-auto">{t("config.skillAutoLearn")}</Label>
          <p className="text-xs text-muted-foreground">{t("config.skillAutoLearnDesc")}</p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <input type="checkbox" id="model-fallback" checked={Boolean(config.model_auto_fallback)}
          onChange={(e) => set("model_auto_fallback", e.target.checked)}
          className="h-4 w-4 rounded border-input" />
        <div>
          <Label htmlFor="model-fallback">{t("config.modelAutoFallback")}</Label>
          <p className="text-xs text-muted-foreground">{t("config.modelAutoFallbackDesc")}</p>
        </div>
      </div>
      <Field label={t("config.workflowRunsDir")}>
        <Input value={String(config.workflow_runs_dir ?? "")} onChange={(e) => set("workflow_runs_dir", e.target.value)}
          placeholder={String(config.default_workflow_runs_dir ?? "")} />
      </Field>
    </div>
  );
}

// ─────────────────────────────────────────────
// Tab: Providers & Models
// ─────────────────────────────────────────────

function ProviderCard({ name, prov, onChange, onDelete, className }: {
  name: string; prov: Provider; onChange: (p: Provider) => void; onDelete: () => void; className?: string;
}) {
  const { t } = useTranslation();
  const set = (k: string, v: unknown) => onChange({ ...prov, [k]: v } as Provider);
  const [addingModel, setAddingModel] = useState(false);
  const [newModelName, setNewModelName] = useState("");
  const testProvider = useTestProvider(name);
  const [testResult, setTestResult] = useState<{ status: string; message?: string; error?: string } | null>(null);

  const handleTest = () => {
    setTestResult(null);
    testProvider.mutate(undefined, {
      onSuccess: (data) => setTestResult(data),
      onError: (err) => setTestResult({ status: "error", error: err.message }),
    });
  };

  const addModel = () => {
    if (!newModelName.trim()) return;
    onChange({ ...prov, models: [...prov.models, { name: newModelName.trim() }] });
    setNewModelName(""); setAddingModel(false);
  };

  return (
    <CollapsibleCard title={name} onDelete={onDelete} className={className}>
      <div className="grid grid-cols-2 gap-3">
        <Field label={t("config.providerType")}>
          <select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            value={prov.type} onChange={(e) => set("type", e.target.value)}>
            <option value="openai">openai</option>
            <option value="deepseek">deepseek</option>
            <option value="anthropic">anthropic</option>
            <option value="ollama">ollama</option>
          </select>
        </Field>
        <Field label={t("config.providerApiUrl")}>
          <Input value={prov.api_url} onChange={(e) => set("api_url", e.target.value)} />
        </Field>
      </div>
      <Field label={t("config.providerApiKey")}>
        <Input value={prov.api_key} onChange={(e) => set("api_key", e.target.value)}
          placeholder={t("config.providerApiKeyPlaceholder")} />
        {prov.api_key?.includes("****") ? (
          <p className="text-xs text-muted-foreground mt-1">{t("config.providerApiKeyMasked")}</p>
        ) : (
          <p className="text-xs text-muted-foreground mt-1">{t("config.secretHint")}</p>
        )}
      </Field>

      {/* Connectivity test */}
      <div className="flex items-center gap-2">
        <Button type="button" variant="outline" size="sm" onClick={handleTest}
          disabled={testProvider.isPending || prov.models.length === 0}>
          {testProvider.isPending ? (
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
          ) : (
            <Zap className="w-3 h-3 mr-1" />
          )}
          {testProvider.isPending ? t("config.testing") : t("config.testConnection")}
        </Button>
        {prov.models.length === 0 && (
          <span className="text-xs text-muted-foreground">{t("config.testNeedsModel")}</span>
        )}
      </div>
      {testResult?.status === "ok" && (
        <div className="flex items-start gap-2 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-sm">
          <CheckCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span className="break-all">{testResult.message}</span>
        </div>
      )}
      {testResult?.status === "error" && (
        <div className="flex items-start gap-2 p-3 rounded-md bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-sm">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span className="break-all">{testResult.error}</span>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <Field label={t("config.imageUrl")}>
          <Input value={prov.image_url ?? ""} onChange={(e) => set("image_url", e.target.value || undefined)} />
        </Field>
        <Field label={t("config.imageModel")}>
          <Input value={prov.image_model ?? ""} onChange={(e) => set("image_model", e.target.value || undefined)} />
        </Field>
        <Field label={t("config.audioUrl")}>
          <Input value={prov.audio_url ?? ""} onChange={(e) => set("audio_url", e.target.value || undefined)} />
        </Field>
        <Field label={t("config.audioModel")}>
          <Input value={prov.audio_model ?? ""} onChange={(e) => set("audio_model", e.target.value || undefined)} />
        </Field>
      </div>

      {/* Rate Limit */}
      <CollapsibleCard title={t("config.rateLimit")}>
        <div className="grid grid-cols-3 gap-3">
          <Field label={t("config.rateLimitRequests")}>
            <NumberInput value={prov.rate_limit?.requests} onChange={(v) =>
              set("rate_limit", { ...prov.rate_limit, requests: v ?? 100, period: prov.rate_limit?.period ?? "1h", margin: prov.rate_limit?.margin ?? 0.1 })} />
          </Field>
          <Field label={t("config.rateLimitPeriod")}>
            <Input value={prov.rate_limit?.period ?? ""} onChange={(e) =>
              set("rate_limit", { ...prov.rate_limit, requests: prov.rate_limit?.requests ?? 100, period: e.target.value, margin: prov.rate_limit?.margin ?? 0.1 })} />
          </Field>
          <Field label={t("config.rateLimitMargin")}>
            <NumberInput value={prov.rate_limit?.margin} onChange={(v) =>
              set("rate_limit", { ...prov.rate_limit, requests: prov.rate_limit?.requests ?? 100, period: prov.rate_limit?.period ?? "1h", margin: v ?? 0.1 })} />
          </Field>
        </div>
      </CollapsibleCard>

      {/* Pricing */}
      <CollapsibleCard title={t("config.pricing")}>
        <div className="grid grid-cols-4 gap-3">
          <Field label={t("config.pricingCurrency")}>
            <Input value={prov.pricing?.currency ?? ""} onChange={(e) =>
              set("pricing", { ...prov.pricing, currency: e.target.value || undefined })} />
          </Field>
          <Field label={t("config.pricingInput")}>
            <NumberInput value={prov.pricing?.input} onChange={(v) => set("pricing", { ...prov.pricing, input: v })} />
          </Field>
          <Field label={t("config.pricingOutput")}>
            <NumberInput value={prov.pricing?.output} onChange={(v) => set("pricing", { ...prov.pricing, output: v })} />
          </Field>
          <Field label={t("config.pricingCachedInput")}>
            <NumberInput value={prov.pricing?.cached_input} onChange={(v) => set("pricing", { ...prov.pricing, cached_input: v })} />
          </Field>
        </div>
      </CollapsibleCard>

      {/* Models */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs font-medium">{t("config.models")}</Label>
          <Button variant="outline" size="sm" onClick={() => setAddingModel(true)}>
            <Plus className="w-3 h-3 mr-1" />{t("config.addModel")}
          </Button>
        </div>
        {addingModel && (
          <div className="flex gap-2">
            <Input value={newModelName} onChange={(e) => setNewModelName(e.target.value)}
              placeholder={t("config.modelName")} className="flex-1"
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addModel(); } }} />
            <Button size="sm" onClick={addModel} disabled={!newModelName.trim()}>
              {t("config.addModel")}
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setAddingModel(false)}>
              <X className="w-3 h-3" />
            </Button>
          </div>
        )}
        {prov.models.map((m, i) => (
          <CollapsibleCard key={m.name} title={m.name}
            onDelete={() => onChange({ ...prov, models: prov.models.filter((_, j) => j !== i) })}>
            <div className="grid grid-cols-2 gap-3">
              <Field label={t("config.temperature")}>
                <NumberInput value={m.options?.temperature} onChange={(v) => {
                  const models = [...prov.models];
                  models[i] = { ...m, options: { ...m.options, temperature: v } };
                  onChange({ ...prov, models });
                }} />
              </Field>
              <Field label={t("config.topP")}>
                <NumberInput value={m.options?.top_p} onChange={(v) => {
                  const models = [...prov.models];
                  models[i] = { ...m, options: { ...m.options, top_p: v } };
                  onChange({ ...prov, models });
                }} />
              </Field>
              <Field label={t("config.maxTokens")}>
                <NumberInput value={m.options?.max_tokens} onChange={(v) => {
                  const models = [...prov.models];
                  models[i] = { ...m, options: { ...m.options, max_tokens: v } };
                  onChange({ ...prov, models });
                }} />
              </Field>
              <Field label={t("config.contextWindow")}>
                <NumberInput value={m.context_window} onChange={(v) => {
                  const models = [...prov.models];
                  models[i] = { ...m, context_window: v };
                  onChange({ ...prov, models });
                }} />
              </Field>
              <Field label={t("config.thinkMode")}>
                <select
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={m.think_mode === true ? "true" : m.think_mode === false ? "false" : "auto"}
                  onChange={(e) => {
                    const models = [...prov.models];
                    const v = e.target.value;
                    models[i] = { ...m, think_mode: v === "auto" ? null : v === "true" };
                    onChange({ ...prov, models });
                  }}
                >
                  <option value="auto">{t("config.thinkModeAuto")}</option>
                  <option value="true">{t("config.thinkModeOn")}</option>
                  <option value="false">{t("config.thinkModeOff")}</option>
                </select>
              </Field>
            </div>
          </CollapsibleCard>
        ))}
      </div>
    </CollapsibleCard>
  );
}

function ProvidersTab({ providers, onChange }: {
  providers: Record<string, Provider>; onChange: (p: Record<string, Provider>) => void;
}) {
  const { t } = useTranslation();
  const [addingProvider, setAddingProvider] = useState(false);
  const [newName, setNewName] = useState("");

  const addProvider = () => {
    if (!newName.trim() || newName in providers) return;
    onChange({ ...providers, [newName.trim()]: { type: "openai", api_url: "", api_key: "", models: [] } });
    setNewName(""); setAddingProvider(false);
  };

  return (
    <div className="space-y-4">
      {Object.keys(providers).length === 0 && (
        <p className="text-sm text-muted-foreground">{t("config.noProviders")}</p>
      )}
      {Object.entries(providers).map(([name, prov], i) => (
        <ProviderCard key={name} name={name} prov={prov}
          className={i % 2 === 1 ? "bg-muted/30" : ""}
          onChange={(p) => onChange({ ...providers, [name]: p })}
          onDelete={() => { const copy = { ...providers }; delete copy[name]; onChange(copy); }} />
      ))}
      {addingProvider ? (
        <div className="flex gap-2">
          <Input value={newName} onChange={(e) => setNewName(e.target.value)}
            placeholder={t("config.providerName")} className="flex-1"
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addProvider(); } }} />
          <Button size="sm" onClick={addProvider} disabled={!newName.trim()}>
            {t("config.addProvider")}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setAddingProvider(false)}>
            <X className="w-3 h-3" />
          </Button>
        </div>
      ) : (
        <Button variant="outline" onClick={() => setAddingProvider(true)}>
          <Plus className="w-4 h-4 mr-1" />{t("config.addProvider")}
        </Button>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Tab: MCP Servers
// ─────────────────────────────────────────────

function MCPServerCard({ name, server, onChange, onDelete, className }: {
  name: string; server: MCPServer; onChange: (s: MCPServer) => void; onDelete: () => void; className?: string;
}) {
  const { t } = useTranslation();
  const isStdio = server.transport === "stdio";

  return (
    <CollapsibleCard title={name} onDelete={onDelete} className={className}>
      <Field label={t("config.transport")}>
        <select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={server.transport} onChange={(e) => onChange({ ...server, transport: e.target.value })}>
          <option value="stdio">stdio</option>
          <option value="streamable_http">streamable_http</option>
          <option value="sse">sse</option>
          <option value="websocket">websocket</option>
        </select>
      </Field>
      {isStdio ? (
        <>
          <Field label={t("config.command")}>
            <Input value={server.command ?? ""} onChange={(e) => onChange({ ...server, command: e.target.value })} />
          </Field>
          <Field label={t("config.args")}>
            <TagInput tags={server.args ?? []} onChange={(v) => onChange({ ...server, args: v })}
              placeholder={t("config.argsPlaceholder")} addLabel={t("config.addArg")} />
          </Field>
          <Field label={t("config.env")}>
            <KVEditor entries={server.env ?? {}} onChange={(v) => onChange({ ...server, env: v })}
              keyLabel={t("config.envKey")} valueLabel={t("config.envValue")} addLabel={t("config.addEnv")} />
          </Field>
        </>
      ) : (
        <>
          <Field label={t("config.url")}>
            <Input value={server.url ?? ""} onChange={(e) => onChange({ ...server, url: e.target.value })} />
          </Field>
          <Field label={t("config.headers")}>
            <KVEditor entries={server.headers ?? {}} onChange={(v) => onChange({ ...server, headers: v })}
              keyLabel={t("config.headerKey")} valueLabel={t("config.headerValue")} addLabel={t("config.addHeader")} />
          </Field>
        </>
      )}
    </CollapsibleCard>
  );
}

function MCPServersTab({ servers, onChange }: {
  servers: Record<string, MCPServer>; onChange: (s: Record<string, MCPServer>) => void;
}) {
  const { t } = useTranslation();
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState("");

  const addServer = () => {
    if (!newName.trim() || newName in servers) return;
    onChange({ ...servers, [newName.trim()]: { transport: "stdio", command: "" } });
    setNewName(""); setAdding(false);
  };

  return (
    <div className="space-y-4">
      {Object.keys(servers).length === 0 && (
        <p className="text-sm text-muted-foreground">{t("config.noServers")}</p>
      )}
      {Object.entries(servers).map(([name, srv], i) => (
        <MCPServerCard key={name} name={name} server={srv}
          className={i % 2 === 1 ? "bg-muted/30" : ""}
          onChange={(s) => onChange({ ...servers, [name]: s })}
          onDelete={() => { const copy = { ...servers }; delete copy[name]; onChange(copy); }} />
      ))}
      {adding ? (
        <div className="flex gap-2">
          <Input value={newName} onChange={(e) => setNewName(e.target.value)}
            placeholder={t("config.serverName")} className="flex-1"
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addServer(); } }} />
          <Button size="sm" onClick={addServer} disabled={!newName.trim()}>
            {t("config.addServer")}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setAdding(false)}>
            <X className="w-3 h-3" />
          </Button>
        </div>
      ) : (
        <Button variant="outline" onClick={() => setAdding(true)}>
          <Plus className="w-4 h-4 mr-1" />{t("config.addServer")}
        </Button>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Tab: Gateway
// ─────────────────────────────────────────────

function GatewayTab({ gateway, onChange }: {
  gateway: Gateway | null; onChange: (g: Gateway) => void;
}) {
  const { t } = useTranslation();
  if (!gateway) return <p className="text-sm text-muted-foreground">{t("config.noGateway")}</p>;

  const set = (k: string, v: unknown) => onChange({ ...gateway, [k]: v } as Gateway);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <Field label={t("config.gwHost")}>
          <Input value={gateway.host} onChange={(e) => set("host", e.target.value)} />
        </Field>
        <Field label={t("config.gwPort")}>
          <NumberInput value={gateway.port} onChange={(v) => set("port", v ?? 8321)} />
        </Field>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Field label={t("config.gwAdminUsername")}>
          <Input value={gateway.admin_username} onChange={(e) => set("admin_username", e.target.value)} />
        </Field>
        <Field label={t("config.gwAdminPassword")}>
          <Input value={gateway.admin_password} onChange={(e) => set("admin_password", e.target.value)}
            placeholder={t("config.providerApiKeyPlaceholder")} />
          {gateway.admin_password?.includes("****") ? (
            <p className="text-xs text-muted-foreground mt-1">{t("config.providerApiKeyMasked")}</p>
          ) : (
            <p className="text-xs text-muted-foreground mt-1">{t("config.secretHint")}</p>
          )}
        </Field>
      </div>
      <Field label={t("config.gwLogFile")}>
        <Input value={gateway.log_file} onChange={(e) => set("log_file", e.target.value)} />
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label={t("config.gwLogMaxBytes")}>
          <NumberInput value={gateway.log_max_bytes} onChange={(v) => set("log_max_bytes", v ?? 10485760)} />
        </Field>
        <Field label={t("config.gwLogBackupCount")}>
          <NumberInput value={gateway.log_backup_count} onChange={(v) => set("log_backup_count", v ?? 5)} />
        </Field>
      </div>

    </div>
  );
}

// ─────────────────────────────────────────────
// Tab: Platforms
// ─────────────────────────────────────────────

function PlatformsTab({ platforms, adapterTypes, onChange }: {
  platforms: Record<string, PlatformAdapter>;
  adapterTypes: string[];
  onChange: (p: Record<string, PlatformAdapter>) => void;
}) {
  const { t } = useTranslation();
  const { data: adapterStatuses } = useAdapters();
  const [healthResults, setHealthResults] = useState<Record<string, { status: string; error?: string }>>({});
  const [checkingHealth, setCheckingHealth] = useState<Record<string, boolean>>({});
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState("");
  const [newType, setNewType] = useState(adapterTypes[0] ?? "");
  const [customType, setCustomType] = useState("");

  const statusMap: Record<string, { state: string; error?: string }> = {};
  if (Array.isArray(adapterStatuses)) {
    for (const a of adapterStatuses) {
      statusMap[a.name as string] = { state: a.state as string, error: a.error as string };
    }
  }

  const updatePlatform = (name: string, patch: Partial<PlatformAdapter>) => {
    onChange({ ...platforms, [name]: { ...platforms[name], ...patch } });
  };

  const deletePlatform = (name: string) => {
    const copy = { ...platforms };
    delete copy[name];
    onChange(copy);
  };

  const addPlatform = () => {
    const name = newName.trim();
    const type = newType === "__custom__" ? customType.trim() : newType;
    if (!name || !type || platforms[name]) return;
    onChange({ ...platforms, [name]: { type, token: "", app_token: "" } });
    setNewName("");
    setNewType(adapterTypes[0] ?? "");
    setCustomType("");
    setAdding(false);
  };

  const checkAdapterHealth = async (name: string) => {
    setCheckingHealth((prev) => ({ ...prev, [name]: true }));
    try {
      const res = await fetch(`/admin/api/adapters/${encodeURIComponent(name)}/check`, {
        method: "POST", credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        setHealthResults((prev) => ({ ...prev, [name]: { status: data.status, error: data.error } }));
      }
    } catch { /* ignore */ }
    setCheckingHealth((prev) => ({ ...prev, [name]: false }));
  };

  const statusBadge = (name: string) => {
    const hr = healthResults[name];
    const s = hr || statusMap[name];
    if (!s) return null;
    const isActive = hr ? hr.status === "active" : (s.state === "connected" || s.state === "running");
    const label = isActive ? "active" : "inactive";
    return (
      <Badge variant={isActive ? "default" : "secondary"} className={`text-[10px] px-1.5 py-0 ${isActive ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : ""}`}>
        {label}
      </Badge>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">{t("config.platforms")}</Label>
        {!adding && (
          <Button variant="outline" size="sm" onClick={() => setAdding(true)}>
            <Plus className="w-3 h-3 mr-1" /> {t("config.addPlatform")}
          </Button>
        )}
      </div>

      {adding && (
        <Card>
          <CardContent className="pt-4 space-y-3">
            <Field label={t("config.platformName")}>
              <Input value={newName} onChange={(e) => setNewName(e.target.value)}
                placeholder={t("config.platformName")}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addPlatform(); } }} />
            </Field>
            <Field label={t("config.platformType")}>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={newType}
                onChange={(e) => setNewType(e.target.value)}
              >
                {adapterTypes.map((at) => (
                  <option key={at} value={at}>{at}</option>
                ))}
                <option value="__custom__">{t("config.platformCustomType")}</option>
              </select>
            </Field>
            {newType === "__custom__" && (
              <Field label={t("config.platformCustomType")}>
                <Input value={customType} onChange={(e) => setCustomType(e.target.value)}
                  placeholder="my.module.MyAdapter" />
              </Field>
            )}
            <div className="flex gap-2 justify-end">
              <Button variant="ghost" size="sm" onClick={() => setAdding(false)}>
                <X className="w-3 h-3 mr-1" /> {t("common.cancel")}
              </Button>
              <Button size="sm" onClick={addPlatform}
                disabled={!newName.trim() || (newType === "__custom__" ? !customType.trim() : !newType)}>
                <Plus className="w-3 h-3 mr-1" /> {t("config.addPlatform")}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {Object.keys(platforms).length === 0 && !adding && (
        <p className="text-sm text-muted-foreground">{t("config.noPlatforms")}</p>
      )}

      {Object.entries(platforms).map(([pname, plat]) => {
        const isBuiltin = adapterTypes.includes(plat.type);
        return (
          <CollapsibleCard key={pname} title={pname}
            onDelete={() => deletePlatform(pname)}>
            <div className="flex items-center gap-2 -mt-1 mb-2">
              <Badge variant="outline" className="text-[10px]">{plat.type}</Badge>
              {statusBadge(pname)}
              <Button variant="ghost" size="sm" className="h-5 px-1.5 text-[10px]"
                onClick={() => checkAdapterHealth(pname)}
                disabled={checkingHealth[pname]}>
                <RotateCw className={`w-3 h-3 mr-0.5 ${checkingHealth[pname] ? "animate-spin" : ""}`} />
                {t("config.checkHealth")}
              </Button>
            </div>
            <Field label={t("config.platformType")}>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={isBuiltin ? plat.type : "__custom__"}
                onChange={(e) => {
                  if (e.target.value === "__custom__") {
                    updatePlatform(pname, { type: "" });
                  } else {
                    updatePlatform(pname, { type: e.target.value });
                  }
                }}
              >
                {adapterTypes.map((at) => (
                  <option key={at} value={at}>{at}</option>
                ))}
                <option value="__custom__">{t("config.platformCustomType")}</option>
              </select>
              {!isBuiltin && (
                <Input className="mt-1" value={plat.type} onChange={(e) => updatePlatform(pname, { type: e.target.value })}
                  placeholder="my.module.MyAdapter" />
              )}
            </Field>
            <Field label={t("config.platformToken")}>
              <Input value={plat.token} onChange={(e) => updatePlatform(pname, { token: e.target.value })}
                placeholder={t("config.providerApiKeyPlaceholder")} />
              {plat.token?.includes("****") ? (
                <p className="text-xs text-muted-foreground mt-1">{t("config.providerApiKeyMasked")}</p>
              ) : (
                <p className="text-xs text-muted-foreground mt-1">{t("config.secretHint")}</p>
              )}
            </Field>
            <Field label={t("config.platformAppToken")}>
              <Input value={plat.app_token} onChange={(e) => updatePlatform(pname, { app_token: e.target.value })}
                placeholder={t("config.providerApiKeyPlaceholder")} />
              {plat.app_token?.includes("****") ? (
                <p className="text-xs text-muted-foreground mt-1">{t("config.providerApiKeyMasked")}</p>
              ) : (
                <p className="text-xs text-muted-foreground mt-1">{t("config.secretHint")}</p>
              )}
            </Field>
            <Field label={t("config.platformHomeChannel")}>
              <Input value={plat.home_channel ?? ""} onChange={(e) =>
                updatePlatform(pname, { home_channel: e.target.value || undefined })} />
            </Field>
            <Field label={t("config.platformAllowedUsers")}>
              <TagInput tags={plat.allowed_users ?? []}
                onChange={(v) => updatePlatform(pname, { allowed_users: v.length ? v : undefined })}
                placeholder={t("config.addUserPlaceholder")} addLabel={t("config.addUser")} />
            </Field>
          </CollapsibleCard>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────
// Tab: Session
// ─────────────────────────────────────────────

function SessionTab({ session, onChange }: {
  session: Session; onChange: (s: Session) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="space-y-4">
      <Field label={t("config.maxSessions")}>
        <NumberInput value={session.max_sessions} onChange={(v) => onChange({ ...session, max_sessions: v ?? 300 })} />
      </Field>
      <Field label={t("config.maxTurnsPerSession")}>
        <NumberInput value={session.max_turns_per_session} onChange={(v) => onChange({ ...session, max_turns_per_session: v ?? 100 })} />
      </Field>
      <Field label={t("config.maxAgeDays")}>
        <NumberInput value={session.max_age_days} onChange={(v) => onChange({ ...session, max_age_days: v ?? 365 })} />
      </Field>
    </div>
  );
}

// ─────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────

export default function ConfigPage() {
  const { t } = useTranslation();
  const { data: rawConfig, isLoading } = useConfig();
  const { data: modelsData } = useModels();
  const saveConfig = useSaveConfig();

  const [config, setConfig] = useState<Config>({});
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (rawConfig) { setConfig(rawConfig); setDirty(false); }
  }, [rawConfig]);

  const update = useCallback((c: Config) => { setConfig(c); setDirty(true); }, []);
  const setProviders = useCallback((p: Record<string, Provider>) => {
    setConfig((prev) => ({ ...prev, providers: p })); setDirty(true);
  }, []);
  const setServers = useCallback((s: Record<string, MCPServer>) => {
    setConfig((prev) => ({ ...prev, mcpServers: s })); setDirty(true);
  }, []);
  const setGateway = useCallback((g: Gateway) => {
    setConfig((prev) => ({ ...prev, gateway: g })); setDirty(true);
  }, []);
  const setSession = useCallback((s: Session) => {
    setConfig((prev) => ({ ...prev, session: s })); setDirty(true);
  }, []);
  const setPlatforms = useCallback((p: Record<string, PlatformAdapter>) => {
    setConfig((prev) => {
      const gw = (prev.gateway as Gateway) ?? { host: "127.0.0.1", port: 8321, admin_username: "", admin_password: "", log_file: "", log_max_bytes: 10485760, log_backup_count: 5, platforms: {} };
      return { ...prev, gateway: { ...gw, platforms: p } };
    });
    setDirty(true);
  }, []);

  const restartGateway = useRestartGateway();
  const [restarting, setRestarting] = useState(false);
  const [confirmRestart, setConfirmRestart] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  const handleSave = () => {
    saveConfig.mutate(config, { onSuccess: () => setDirty(false) });
  };

  const handleRestart = () => {
    setRestarting(true);
    restartGateway.mutate(undefined, {
      onSuccess: () => {
        pollRef.current = setInterval(async () => {
          try {
            const res = await fetch("/admin/api/adapters");
            if (res.ok) {
              if (pollRef.current) clearInterval(pollRef.current);
              window.location.reload();
            }
          } catch { /* gateway still restarting */ }
        }, 1000);
        timeoutRef.current = setTimeout(() => {
          if (pollRef.current) clearInterval(pollRef.current);
          setRestarting(false);
        }, 15000);
      },
      onError: () => setRestarting(false),
    });
  };

  const modelList = (modelsData as { models?: string[] })?.models ?? [];

  if (isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("config.title")}</h1>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => setConfirmRestart(true)} disabled={restarting}>
            <RotateCw className={`w-4 h-4 mr-1 ${restarting ? "animate-spin" : ""}`} />
            {restarting ? t("config.restarting") : t("config.restart")}
          </Button>
          <Button size="sm" onClick={handleSave} disabled={!dirty || saveConfig.isPending}>
            {saveConfig.isPending ? (
              <Loader2 className="w-4 h-4 mr-1 animate-spin" />
            ) : (
              <Save className="w-4 h-4 mr-1" />
            )}
            {saveConfig.isPending ? t("config.saving") : t("config.save")}
          </Button>
        </div>
      </div>

      {saveConfig.isSuccess && !dirty && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-sm">
          <CheckCircle className="w-4 h-4 shrink-0" />
          {t("config.saved")}
        </div>
      )}
      {saveConfig.isError && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-sm">
          <XCircle className="w-4 h-4 shrink-0" />
          {t("config.saveError", { error: saveConfig.error?.message ?? "Unknown" })}
        </div>
      )}

      <Tabs defaultValue="general">
        <TabsList>
          <TabsTrigger value="general">{t("config.general")}</TabsTrigger>
          <TabsTrigger value="providers">{t("config.providers")}</TabsTrigger>
          <TabsTrigger value="mcpServers">{t("config.mcpServers")}</TabsTrigger>
          <TabsTrigger value="gateway">{t("config.gateway")}</TabsTrigger>
          <TabsTrigger value="platforms">{t("config.platformsTab")}</TabsTrigger>
          <TabsTrigger value="session">{t("config.session")}</TabsTrigger>
        </TabsList>
        <TabsContent value="general">
          <GeneralTab config={config} setConfig={update} modelList={modelList} />
        </TabsContent>
        <TabsContent value="providers">
          <ProvidersTab providers={(config.providers ?? {}) as Record<string, Provider>} onChange={setProviders} />
        </TabsContent>
        <TabsContent value="mcpServers">
          <MCPServersTab servers={(config.mcpServers ?? {}) as Record<string, MCPServer>} onChange={setServers} />
        </TabsContent>
        <TabsContent value="gateway">
          <GatewayTab gateway={(config.gateway as Gateway) ?? null} onChange={setGateway} />
        </TabsContent>
        <TabsContent value="platforms">
          <PlatformsTab
            platforms={((config.gateway as Gateway)?.platforms ?? {}) as Record<string, PlatformAdapter>}
            adapterTypes={(config.adapter_types as string[]) ?? []}
            onChange={setPlatforms} />
        </TabsContent>
        <TabsContent value="session">
          <SessionTab session={(config.session as Session) ?? { max_sessions: 300, max_turns_per_session: 100, max_age_days: 365 }} onChange={setSession} />
        </TabsContent>
      </Tabs>

      <ConfirmDialog
        open={confirmRestart}
        title={t("config.restartConfirmTitle")}
        description={t("config.restartConfirmDesc")}
        onCancel={() => setConfirmRestart(false)}
        onConfirm={() => { setConfirmRestart(false); handleRestart(); }}
      />
    </div>
  );
}
