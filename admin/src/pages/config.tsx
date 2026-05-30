import { useState, useEffect, useCallback, useRef } from "react";
import {
  Save, Loader2, CheckCircle, XCircle, Trash2, X, ChevronDown, ChevronRight, RotateCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useConfig, useSaveConfig, useRestartGateway } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

type Config = Record<string, unknown>;
type Gateway = {
  host: string; port: number; admin_username: string; admin_password: string;
  log_file: string; log_max_bytes: number; log_backup_count: number;
  api_key?: string;
};
type Session = { max_sessions: number; max_turns_per_session: number; max_age_days: number };

function _TagInput({ tags, onChange, placeholder, addLabel, disabled }: {
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

function _KVEditor({ entries, onChange, keyLabel, valueLabel, addLabel }: {
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

function _CollapsibleCard({ title, defaultOpen, onDelete, className, children }: {
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

function GeneralTab({ config, setConfig }: {
  config: Config; setConfig: (c: Config) => void;
}) {
  const { t } = useTranslation();
  const set = (k: string, v: unknown) => setConfig({ ...config, [k]: v });

  return (
    <div className="space-y-4">
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
  const saveConfig = useSaveConfig();

  const [config, setConfig] = useState<Config>({});
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (rawConfig) { setConfig(rawConfig); setDirty(false); }
  }, [rawConfig]);

  const update = useCallback((c: Config) => { setConfig(c); setDirty(true); }, []);
  const setGateway = useCallback((g: Gateway) => {
    setConfig((prev) => ({ ...prev, gateway: g })); setDirty(true);
  }, []);
  const setSession = useCallback((s: Session) => {
    setConfig((prev) => ({ ...prev, session: s })); setDirty(true);
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
          <TabsTrigger value="gateway">{t("config.gateway")}</TabsTrigger>
          <TabsTrigger value="session">{t("config.session")}</TabsTrigger>
        </TabsList>
        <TabsContent value="general">
          <GeneralTab config={config} setConfig={update} />
        </TabsContent>
        <TabsContent value="gateway">
          <GatewayTab gateway={(config.gateway as Gateway) ?? null} onChange={setGateway} />
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
