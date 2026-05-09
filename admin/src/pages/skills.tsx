import { Link } from "react-router";
import { Plus } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useSkills } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

export default function SkillsPage() {
  const { t } = useTranslation();
  const { data, isLoading } = useSkills();
  if (isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;

  const skills = (data ?? []) as Array<{
    name: string;
    description?: string;
    source?: string;
  }>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("skills.title")}</h1>
        <Link to="/admin/skills/new">
          <Button size="sm">
            <Plus className="w-4 h-4 mr-1" />
            {t("skills.createSkill")}
          </Button>
        </Link>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {skills.map((s) => (
          <Link
            key={s.name}
            to={`/admin/skills/${encodeURIComponent(s.name)}`}
            className="block"
          >
            <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center justify-between">
                  {s.name}
                  {s.source && <Badge variant="secondary">{s.source}</Badge>}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  {s.description || t("common.noDescription")}
                </p>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
