import { useState } from "react";
import { useParams, useNavigate } from "react-router";
import { ArrowLeft, Plus, Trash2, Play, FlaskConical, MessageSquare } from "lucide-react";
import { useLab, useUpdateLab, useLabGoldenTests, useCreateGoldenTest, useDeleteGoldenTest, useRunGoldenTests, type TestRunResult } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { LabHeader } from "@/components/lab/LabHeader";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";

export default function LabDetailPage() {
  const { t } = useTranslation();
  const { labId } = useParams<{ labId: string }>();
  const navigate = useNavigate();
  const { data: labDetail, isLoading } = useLab(labId);
  const { data: tests, isLoading: testsLoading } = useLabGoldenTests(labId);
  const updateLab = useUpdateLab(labId!);
  const createTest = useCreateGoldenTest(labId!);
  const deleteTest = useDeleteGoldenTest(labId!);
  const runTests = useRunGoldenTests(labId!);

  const [editTitle, setEditTitle] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [editing, setEditing] = useState(false);
  const [showAddTest, setShowAddTest] = useState(false);
  const [testName, setTestName] = useState("");
  const [testInput, setTestInput] = useState("");
  const [testExpected, setTestExpected] = useState("");
  const [testResults, setTestResults] = useState<TestRunResult[] | null>(null);

  if (labDetail && !editing && !editTitle) {
    setEditTitle(labDetail.lab.title);
    setEditDesc(labDetail.lab.description || "");
  }

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">{t("common.loading")}</p>;
  }

  if (!labDetail) {
    return (
      <div className="flex flex-col items-center py-16">
        <h2 className="text-lg font-medium">{t("labs.notFound")}</h2>
        <Button className="mt-4" onClick={() => navigate("/admin/labs")}>
          <ArrowLeft className="w-4 h-4 mr-1" /> {t("common.back")}
        </Button>
      </div>
    );
  }

  const lab = labDetail.lab;

  return (
    <div className="space-y-6">
      {editing ? (
        <div className="space-y-4">
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <button onClick={() => navigate("/admin/labs")} className="hover:text-foreground transition-colors">
              {t("labs.title")}
            </button>
            <span className="mx-0.5">›</span>
            <span className="text-foreground">{lab.title}</span>
          </div>
          <Input
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            className="text-xl font-semibold"
          />
          <Textarea
            value={editDesc}
            onChange={(e) => setEditDesc(e.target.value)}
            rows={3}
          />
          <div className="flex gap-2">
            <Button size="sm" onClick={async () => {
              await updateLab.mutateAsync({ title: editTitle, description: editDesc });
              setEditing(false);
            }} disabled={updateLab.isPending}>
              {updateLab.isPending ? t("labs.savingLab") : t("labs.saveLab")}
            </Button>
            <Button size="sm" variant="outline" onClick={() => setEditing(false)}>{t("common.cancel")}</Button>
          </div>
        </div>
      ) : (
        <div>
          <LabHeader
            title={lab.title}
            description={lab.description}
            breadcrumb={
              <>
                <button onClick={() => navigate("/admin/labs")} className="hover:text-foreground transition-colors">
                  {t("labs.title")}
                </button>
                <span className="mx-0.5">›</span>
                <span className="text-foreground">{lab.title}</span>
              </>
            }
          />
          <div className="flex gap-2 mt-3">
            <Button variant="outline" size="sm" onClick={() => setEditing(true)}>{t("labs.editLab")}</Button>
            <Button size="sm" onClick={() => navigate(`/admin/labs/${labId}/chat`)}>
              <MessageSquare className="w-4 h-4 mr-1" /> {t("lab.chat")}
            </Button>
          </div>
        </div>
      )}

      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium flex items-center gap-2">
            <FlaskConical className="w-5 h-5" /> {t("lab.goldenTests")}
          </h2>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setShowAddTest(true)}>
              <Plus className="w-3.5 h-3.5 mr-1" /> {t("lab.addTest")}
            </Button>
            <Button
              size="sm"
              onClick={async () => {
                const result = await runTests.mutateAsync();
                setTestResults(result.results);
              }}
              disabled={tests?.length === 0 || runTests.isPending}
            >
              <Play className="w-3.5 h-3.5 mr-1" />
              {runTests.isPending ? t("common.running") : t("lab.runTests")}
            </Button>
          </div>
        </div>

        {testsLoading && <p className="text-sm text-muted-foreground">{t("common.loading")}</p>}

        {!testsLoading && tests && tests.length === 0 && (
          <p className="text-sm text-muted-foreground py-4 text-center">{t("lab.noTests")}</p>
        )}

        <div className="space-y-2">
          {(tests || []).map((test) => {
            const result = testResults?.find((r) => r.test_id === test.id);
            return (
              <div key={test.id} className="flex items-start gap-3 p-3 rounded-md border">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{test.name}</span>
                    {result && (
                      <Badge variant="secondary" className="text-[10px]">
                        {t("lab.passing")}
                      </Badge>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                    <strong>{t("lab.testInput")}:</strong> {test.input}
                  </p>
                  {result && (
                    <div className="mt-2 space-y-1 text-xs">
                      <p><strong>{t("lab.expectedOutput")}:</strong> {test.expected_output}</p>
                      <p className="text-muted-foreground"><strong>{t("lab.actualOutput")}:</strong> {result.actual_output.slice(0, 300)}</p>
                    </div>
                  )}
                </div>
                <button
                  className="p-1 hover:bg-destructive/10 rounded shrink-0"
                  onClick={() => deleteTest.mutate(test.id)}
                >
                  <Trash2 className="w-3.5 h-3.5 text-destructive" />
                </button>
              </div>
            );
          })}
        </div>
      </Card>

      <Dialog open={showAddTest} onOpenChange={setShowAddTest}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("lab.addTest")}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">{t("lab.testName")}</label>
              <Input value={testName} onChange={(e) => setTestName(e.target.value)} placeholder="e.g. Order with discount" />
            </div>
            <div>
              <label className="text-sm font-medium">{t("lab.testInput")}</label>
              <Textarea value={testInput} onChange={(e) => setTestInput(e.target.value)} rows={2} placeholder="Customer wants to return..." />
            </div>
            <div>
              <label className="text-sm font-medium">{t("lab.testExpectedOutput")}</label>
              <Textarea value={testExpected} onChange={(e) => setTestExpected(e.target.value)} rows={3} placeholder="Agent should: check return window..." />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowAddTest(false)}>{t("common.cancel")}</Button>
              <Button
                onClick={async () => {
                  if (!testName.trim()) return;
                  await createTest.mutateAsync({
                    name: testName.trim(),
                    input: testInput,
                    expected_output: testExpected,
                  });
                  setShowAddTest(false);
                  setTestName("");
                  setTestInput("");
                  setTestExpected("");
                }}
                disabled={!testName.trim() || createTest.isPending}
              >
                {t("common.add")}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
