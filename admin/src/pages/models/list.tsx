import { useState } from "react";
import { Plus, Trash2, Bot, Pencil } from "lucide-react";
import {
  useModels, useCreateModel, useUpdateModel, useDeleteModel,
  useProviders, useCreateProvider, useEndpoints, useCreateEndpoint,
  useKeys, useCreateKey,
  type ModelInfo, type ModelProvider, type ModelEndpoint,
} from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/confirm-dialog";

const PROVIDER_COLORS: Record<string, string> = {
  openai: "bg-amber-500",
  anthropic: "bg-orange-500",
  deepseek: "bg-blue-500",
  ollama: "bg-gray-500",
};

const MODALITY_COLORS: Record<string, string> = {
  text: "bg-emerald-500",
  image: "bg-purple-500",
  audio: "bg-red-500",
  video: "bg-pink-500",
};

const CAPABILITY_OPTIONS = [
  "text_to_text", "image_to_text", "text_to_image",
  "text_to_audio", "audio_to_text", "text_to_video",
  "video_to_text", "tool_calling",
];

function getModalities(capabilities: string[]): string[] {
  const mods: string[] = [];
  const textCaps = ["text_to_text"];
  const imageCaps = ["image_to_text", "text_to_image"];
  const audioCaps = ["audio_to_text", "text_to_audio"];
  const videoCaps = ["video_to_text", "text_to_video"];
  if (capabilities.some((c) => textCaps.includes(c))) mods.push("text");
  if (capabilities.some((c) => imageCaps.includes(c))) mods.push("image");
  if (capabilities.some((c) => audioCaps.includes(c))) mods.push("audio");
  if (capabilities.some((c) => videoCaps.includes(c))) mods.push("video");
  return mods;
}

function timeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMin = Math.floor((now - then) / 60000);
  if (diffMin < 1) return "Just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

function OptionsTable({ options, onChange }: {
  options: Record<string, string>;
  onChange: (o: Record<string, string>) => void;
}) {
  const entries = Object.entries(options);
  const set = (k: string, v: string) => onChange({ ...options, [k]: v });
  const remove = (k: string) => {
    const copy = { ...options };
    delete copy[k];
    onChange(copy);
  };
  const add = () => onChange({ ...options, "": "" });

  return (
    <div className="space-y-1">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted-foreground">
            <th className="text-left font-medium py-1">Key</th>
            <th className="text-left font-medium py-1">Value</th>
            <th className="w-8"></th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([k, v], i) => (
            <tr key={i}>
              <td className="pr-1 py-0.5">
                <Input
                  value={k}
                  onChange={(e) => { remove(k); set(e.target.value, v); }}
                  className="h-7 text-xs"
                  placeholder="key"
                />
              </td>
              <td className="pr-1 py-0.5">
                <Input
                  value={v}
                  onChange={(e) => set(k, e.target.value)}
                  className="h-7 text-xs"
                  placeholder="value"
                />
              </td>
              <td className="py-0.5">
                <button onClick={() => remove(k)} className="text-destructive hover:text-destructive/80">
                  <Trash2 className="w-3 h-3" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Button variant="ghost" size="sm" className="text-xs h-6" onClick={add}>
        + Add option
      </Button>
    </div>
  );
}

function ModelForm({ model, onChange, providers, t }: {
  model: Partial<ModelInfo> & { provider_id?: string; endpoint_id?: string };
  onChange: (m: Partial<ModelInfo>) => void;
  providers: ModelProvider[];
  t: (key: string, params?: Record<string, string>) => string;
}) {
  const set = (k: string, v: unknown) => onChange({ ...model, [k]: v });
  const { data: endpoints } = useEndpoints(model.provider_id || "");

  const handleOptionsChange = (opts: Record<string, string>) => {
    const cleaned: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(opts)) {
      if (k.trim()) {
        const num = Number(v);
        cleaned[k.trim()] = isNaN(num) ? v : num;
      }
    }
    set("options", cleaned);
  };

  const optionsStr: Record<string, string> = {};
  if (model.options) {
    for (const [k, v] of Object.entries(model.options)) {
      optionsStr[k] = String(v);
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm font-medium">{t("models.providerLabel")} *</label>
        <select
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={model.provider_id || ""}
          onChange={(e) => set("provider_id", e.target.value)}
          autoFocus
        >
          <option value="">{t("models.newProvider")}</option>
          {providers.map((p) => (
            <option key={p.id} value={p.id}>{p.name} ({p.type})</option>
          ))}
        </select>
      </div>
      {model.provider_id && (
        <div>
          <label className="text-sm font-medium">{t("models.endpointLabel")} *</label>
          <select
            className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            value={model.endpoint_id || ""}
            onChange={(e) => set("endpoint_id", e.target.value)}
          >
            <option value="">{t("models.newEndpoint")}</option>
            {(endpoints || []).map((ep) => (
              <option key={ep.id} value={ep.id}>{ep.base_url}</option>
            ))}
          </select>
        </div>
      )}
      <div>
        <label className="text-sm font-medium">{t("models.nameLabel")} *</label>
        <Input
          value={model.name || ""}
          onChange={(e) => set("name", e.target.value)}
          placeholder={t("models.namePlaceholder")}
        />
      </div>
      <div>
        <label className="text-sm font-medium">{t("models.capabilities")}</label>
        <div className="flex flex-wrap gap-2 mt-1">
          {CAPABILITY_OPTIONS.map((cap) => {
            const checked = (model.capabilities || []).includes(cap);
            return (
              <label key={cap} className="flex items-center gap-1 text-xs">
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => {
                    const caps = checked
                      ? (model.capabilities || []).filter((c) => c !== cap)
                      : [...(model.capabilities || []), cap];
                    set("capabilities", caps);
                  }}
                  className="h-3.5 w-3.5"
                />
                {cap}
              </label>
            );
          })}
        </div>
      </div>
      <div>
        <label className="text-sm font-medium">{t("models.optionsLabel")}</label>
        <OptionsTable options={optionsStr} onChange={handleOptionsChange} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-sm font-medium">{t("models.thinkMode")}</label>
          <select
            className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            value={model.think_mode === null || model.think_mode === undefined ? "auto" : String(model.think_mode)}
            onChange={(e) => {
              const v = e.target.value;
              set("think_mode", v === "auto" ? null : Number(v));
            }}
          >
            <option value="auto">{t("models.thinkModeAuto")}</option>
            <option value="1">{t("models.thinkModeOn")}</option>
            <option value="0">{t("models.thinkModeOff")}</option>
          </select>
        </div>
        <div>
          <label className="text-sm font-medium">{t("models.contextWindow")}</label>
          <Input
            type="number"
            value={model.context_window ?? ""}
            onChange={(e) => set("context_window", e.target.value ? Number(e.target.value) : null)}
            placeholder="e.g. 128000"
            className="mt-1"
          />
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2">
        <div>
          <label className="text-xs font-medium">{t("models.pricingCurrency")}</label>
          <Input
            value={model.pricing?.currency ?? ""}
            onChange={(e) => set("pricing", { ...model.pricing, currency: e.target.value || undefined })}
            className="h-7 text-xs mt-0.5"
          />
        </div>
        <div>
          <label className="text-xs font-medium">{t("models.pricingInput")}</label>
          <Input
            type="number"
            value={model.pricing?.input ?? ""}
            onChange={(e) => set("pricing", { ...model.pricing, input: e.target.value ? Number(e.target.value) : undefined })}
            className="h-7 text-xs mt-0.5"
          />
        </div>
        <div>
          <label className="text-xs font-medium">{t("models.pricingOutput")}</label>
          <Input
            type="number"
            value={model.pricing?.output ?? ""}
            onChange={(e) => set("pricing", { ...model.pricing, output: e.target.value ? Number(e.target.value) : undefined })}
            className="h-7 text-xs mt-0.5"
          />
        </div>
        <div>
          <label className="text-xs font-medium">{t("models.pricingCachedInput")}</label>
          <Input
            type="number"
            value={model.pricing?.cached_input ?? ""}
            onChange={(e) => set("pricing", { ...model.pricing, cached_input: e.target.value ? Number(e.target.value) : undefined })}
            className="h-7 text-xs mt-0.5"
          />
        </div>
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<string>("all");

  const { data: models, isLoading } = useModels(activeTab !== "all" ? activeTab : undefined);
  const { data: providers } = useProviders();
  const { data: keys } = useKeys();
  const createModel = useCreateModel();
  const [editTarget, setEditTarget] = useState<ModelInfo | null>(null);
  const updateModel = useUpdateModel(editTarget?.id || "");
  const deleteModel = useDeleteModel();
  const createProvider = useCreateProvider();
  const createEndpoint = useCreateEndpoint("");
  const createKey = useCreateKey();

  const [showCreate, setShowCreate] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<ModelInfo | null>(null);
  const [form, setForm] = useState<Partial<ModelInfo>>({ capabilities: [], options: {} });
  const [showNewProvider, setShowNewProvider] = useState(false);
  const [showNewEndpoint, setShowNewEndpoint] = useState(false);
  const [showNewKey, setShowNewKey] = useState(false);
  const [providerForm, setProviderForm] = useState({ name: "", type: "openai", api_key: "" });
  const [newEndpointUrl, setNewEndpointUrl] = useState("");
  const [newEndpointProvId, setNewEndpointProvId] = useState("");
  const [keyForm, setKeyForm] = useState({ name: "", value: "", description: "" });

  const resetForm = () => {
    setForm({ capabilities: [], options: {} });
    setShowNewProvider(false);
    setShowNewEndpoint(false);
  };

  const handleCreate = async () => {
    if (!form.name?.trim() || !form.provider_id || !form.endpoint_id) return;
    await createModel.mutateAsync({
      name: form.name.trim(),
      provider_id: form.provider_id,
      endpoint_id: form.endpoint_id,
      capabilities: form.capabilities,
      options: form.options,
      think_mode: form.think_mode,
      context_window: form.context_window,
      pricing: form.pricing,
    });
    setShowCreate(false);
    resetForm();
  };

  const handleUpdate = async () => {
    if (!editTarget || !form.name?.trim()) return;
    await updateModel.mutateAsync({
      name: form.name.trim(),
      endpoint_id: form.endpoint_id,
      capabilities: form.capabilities,
      options: form.options,
      think_mode: form.think_mode,
      context_window: form.context_window,
      pricing: form.pricing,
    });
    setEditTarget(null);
    resetForm();
  };

  const openEdit = (m: ModelInfo) => {
    setEditTarget(m);
    setForm({ ...m });
  };

  const handleCreateProvider = async () => {
    if (!providerForm.name.trim()) return;
    const p = await createProvider.mutateAsync({
      name: providerForm.name.trim(),
      type: providerForm.type,
      api_key: providerForm.api_key || undefined,
    });
    setForm({ ...form, provider_id: p.id });
    setShowNewProvider(false);
    setProviderForm({ name: "", type: "openai", api_key: "" });
  };

  const handleCreateEndpoint = async () => {
    if (!newEndpointUrl.trim() || !newEndpointProvId) return;
    const ep = await createEndpoint.mutateAsync({ base_url: newEndpointUrl.trim() });
    setForm({ ...form, endpoint_id: ep.id });
    setShowNewEndpoint(false);
    setNewEndpointUrl("");
  };

  const handleCreateKey = async () => {
    if (!keyForm.name.trim() || !keyForm.value.trim()) return;
    await createKey.mutateAsync({
      name: keyForm.name.trim(),
      value: keyForm.value,
      description: keyForm.description,
    });
    setProviderForm({ ...providerForm, api_key: `{{ key('${keyForm.name.trim()}') }}` });
    setShowNewKey(false);
    setKeyForm({ name: "", value: "", description: "" });
  };

  const tabs = ["all", "text", "image", "audio", "video"];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">{t("models.title")}</h1>
        <Button onClick={() => { resetForm(); setShowCreate(true); }}>
          <Plus className="w-4 h-4 mr-1" />
          {t("models.newModel")}
        </Button>
      </div>

      {isLoading && <p className="text-sm text-muted-foreground">{t("common.loading")}</p>}

      {!isLoading && models && models.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Bot className="w-12 h-12 text-muted-foreground/50 mb-4" />
          <h2 className="text-lg font-medium">{t("models.noModels")}</h2>
          <p className="text-sm text-muted-foreground mt-1 max-w-sm">{t("models.noModelsDesc")}</p>
          <Button className="mt-4" onClick={() => { resetForm(); setShowCreate(true); }}>
            <Plus className="w-4 h-4 mr-1" />
            {t("models.createTitle")}
          </Button>
        </div>
      )}

      {/* Type filter tabs */}
      <div className="flex gap-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
              activeTab === tab
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            {tab === "all" ? t("models.allModels") : t(`models.${tab}Models`)}
            {tab === "all" && models && ` (${models.length})`}
          </button>
        ))}
      </div>

      {/* Card grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {(models || []).map((model) => {
          const provider = (providers || []).find((p) => p.id === model.provider_id);
          const providerName = provider?.name || model.provider_id;
          const modalities = getModalities(model.capabilities || []);
          return (
            <Card
              key={model.id}
              className="p-4 hover:shadow-md transition-shadow cursor-pointer group"
              onClick={() => openEdit(model)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium truncate">{providerName}/{model.name}</h3>
                  <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                    <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded text-white ${PROVIDER_COLORS[provider?.type || ""] || "bg-gray-500"}`}>
                      {providerName}
                    </span>
                    {modalities.map((mod) => (
                      <span key={mod} className={`text-[10px] font-medium px-1.5 py-0.5 rounded text-white ${MODALITY_COLORS[mod] || "bg-gray-400"}`}>
                        {mod}
                      </span>
                    ))}
                    {model.is_default === 1 && (
                      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-primary/20 text-primary">default</span>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between mt-3 pt-3 border-t">
                <span className="text-xs text-muted-foreground">{timeAgo(model.updated_at)}</span>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button className="p-1 hover:bg-muted rounded" onClick={(e) => { e.stopPropagation(); openEdit(model); }}>
                    <Pencil className="w-3.5 h-3.5" />
                  </button>
                  <button className="p-1 hover:bg-destructive/10 rounded text-destructive" onClick={(e) => { e.stopPropagation(); setDeleteTarget(model); }}>
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Create Dialog */}
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("models.createTitle")}</DialogTitle>
          </DialogHeader>

          {/* Provider inline form */}
          {showNewProvider && (
            <div className="border border-primary/30 rounded-lg p-3 space-y-2 bg-primary/5">
              <h4 className="text-sm font-medium">{t("models.newProvider")}</h4>
              <Input
                value={providerForm.name}
                onChange={(e) => setProviderForm({ ...providerForm, name: e.target.value })}
                placeholder={t("models.providerName")}
              />
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={providerForm.type}
                onChange={(e) => setProviderForm({ ...providerForm, type: e.target.value })}
              >
                <option value="openai">openai</option>
                <option value="deepseek">deepseek</option>
                <option value="anthropic">anthropic</option>
                <option value="ollama">ollama</option>
              </select>
              <div>
                <div className="flex gap-1">
                  <select
                    className="flex-1 rounded-md border border-input bg-background px-2 py-1.5 text-sm"
                    value={providerForm.api_key}
                    onChange={(e) => {
                      if (e.target.value === "__new__") { setShowNewKey(true); return; }
                      setProviderForm({ ...providerForm, api_key: e.target.value });
                    }}
                  >
                    <option value="">{t("models.selectKey")}</option>
                    {(keys || []).map((k) => (
                      <option key={k.name} value={`{{ key('${k.name}') }}`}>🔑 {k.name}</option>
                    ))}
                    <option value="__new__">+ {t("models.createKey")}</option>
                  </select>
                </div>
                {showNewKey && (
                  <div className="mt-2 space-y-1.5 border border-green-500/30 rounded-lg p-2 bg-green-500/5">
                    <Input
                      value={keyForm.name}
                      onChange={(e) => setKeyForm({ ...keyForm, name: e.target.value })}
                      placeholder={t("models.keyName")}
                      className="h-7 text-xs"
                    />
                    <Input
                      value={keyForm.value}
                      onChange={(e) => setKeyForm({ ...keyForm, value: e.target.value })}
                      placeholder={t("models.keyValue")}
                      className="h-7 text-xs"
                    />
                    <Input
                      value={keyForm.description}
                      onChange={(e) => setKeyForm({ ...keyForm, description: e.target.value })}
                      placeholder={t("models.keyDescription")}
                      className="h-7 text-xs"
                    />
                    <div className="flex justify-end gap-1">
                      <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => setShowNewKey(false)}>
                        {t("common.cancel")}
                      </Button>
                      <Button size="sm" className="h-6 text-xs" onClick={handleCreateKey} disabled={createKey.isPending}>
                        {t("models.createAndSelect")}
                      </Button>
                    </div>
                  </div>
                )}
              </div>
              <div className="flex justify-end gap-1">
                <Button variant="ghost" size="sm" onClick={() => setShowNewProvider(false)}>{t("common.cancel")}</Button>
                <Button size="sm" onClick={handleCreateProvider} disabled={createProvider.isPending}>
                  {createProvider.isPending ? t("models.saving") : t("common.save")}
                </Button>
              </div>
            </div>
          )}

          <ModelForm model={form} onChange={setForm} providers={providers || []} t={t} />

          {/* New endpoint inline */}
          {showNewEndpoint && (
            <div className="border border-primary/30 rounded-lg p-3 space-y-2 bg-primary/5">
              <h4 className="text-sm font-medium">{t("models.newEndpoint")}</h4>
              <Input
                value={newEndpointUrl}
                onChange={(e) => setNewEndpointUrl(e.target.value)}
                placeholder={t("models.endpointBaseUrl")}
              />
              <div className="flex justify-end gap-1">
                <Button variant="ghost" size="sm" onClick={() => setShowNewEndpoint(false)}>{t("common.cancel")}</Button>
                <Button size="sm" onClick={handleCreateEndpoint}>
                  {t("common.save")}
                </Button>
              </div>
            </div>
          )}

          {!showNewProvider && (
            <Button variant="outline" size="sm" onClick={() => setShowNewProvider(true)}>
              <Plus className="w-3 h-3 mr-1" /> {t("models.newProvider")}
            </Button>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setShowCreate(false)}>{t("common.cancel")}</Button>
            <Button onClick={handleCreate} disabled={!form.name?.trim() || !form.provider_id || !form.endpoint_id || createModel.isPending}>
              {createModel.isPending ? t("models.saving") : t("common.save")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editTarget} onOpenChange={() => setEditTarget(null)}>
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("models.editTitle")}</DialogTitle>
          </DialogHeader>
          <ModelForm model={form} onChange={setForm} providers={providers || []} t={t} />
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setEditTarget(null)}>{t("common.cancel")}</Button>
            <Button onClick={handleUpdate} disabled={!form.name?.trim() || updateModel.isPending}>
              {updateModel.isPending ? t("models.saving") : t("common.save")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={!!deleteTarget}
        onCancel={() => setDeleteTarget(null)}
        title={t("models.deleteTitle")}
        description={t("models.deleteDesc", { name: deleteTarget ? `${deleteTarget.name}` : "" })}
        onConfirm={() => {
          if (deleteTarget) {
            deleteModel.mutate(deleteTarget.id);
            setDeleteTarget(null);
          }
        }}
      />
    </div>
  );
}
