import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "@/i18n";

export function ThemeToggle() {
  const { t } = useTranslation();
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return true;
    return localStorage.getItem("cliver-theme") !== "light";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("cliver-theme", dark ? "dark" : "light");
  }, [dark]);

  return (
    <button
      onClick={() => setDark((d) => !d)}
      className="flex items-center justify-center w-8 h-8 rounded-md hover:bg-sidebar-accent text-muted-foreground hover:text-foreground transition-colors"
      title={dark ? t("theme.switchToLight") : t("theme.switchToDark")}
    >
      {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
    </button>
  );
}
