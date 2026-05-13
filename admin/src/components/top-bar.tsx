import { Globe, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTranslation, type Locale } from "@/i18n";

const LANGUAGES: { value: Locale; label: string }[] = [
  { value: "en", label: "English" },
  { value: "zh", label: "中文" },
];

export function TopBar() {
  const { t, locale, setLocale } = useTranslation();
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return true;
    return localStorage.getItem("cliver-theme") !== "light";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("cliver-theme", dark ? "dark" : "light");
  }, [dark]);

  const currentLang = LANGUAGES.find((l) => l.value === locale) || LANGUAGES[0];

  return (
    <div className="flex items-center justify-end gap-2 px-6 py-2 border-b border-border bg-background/80 backdrop-blur-sm">
      {/* Language dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors">
            <Globe className="w-4 h-4" />
            {currentLang?.label || "EN"}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {LANGUAGES.map((lang) => (
            <DropdownMenuItem
              key={lang.value}
              onClick={() => setLocale(lang.value)}
              className={locale === lang.value ? "font-medium" : ""}
            >
              {lang.label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Theme toggle */}
      <button
        onClick={() => setDark((d) => !d)}
        className="flex items-center justify-center w-8 h-8 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
        title={dark ? t("theme.switchToLight") : t("theme.switchToDark")}
      >
        {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
      </button>
    </div>
  );
}
