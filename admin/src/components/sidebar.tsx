import { NavLink } from "react-router";
import {
  Book,
  FolderOpen,
  ListTodo,
  Key,
  MessageSquare,
  Brain,
  Settings,
  Languages,
} from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { CliverLogo } from "@/components/cliver-logo";
import { useTranslation } from "@/i18n";
import { cn } from "@/lib/utils";

interface NavItem {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  labelKey: string;
}

interface NavSection {
  titleKey: string;
  items: NavItem[];
}

const navSections: NavSection[] = [
  {
    titleKey: "sidebar.section.workspace",
    items: [
      { to: "/admin/notebooks", icon: Book, labelKey: "sidebar.notebooks" },
      { to: "/admin/projects", icon: FolderOpen, labelKey: "sidebar.projects" },
      { to: "/admin/tasks", icon: ListTodo, labelKey: "sidebar.tasks" },
    ],
  },
  {
    titleKey: "sidebar.section.system",
    items: [
      { to: "/admin/keys", icon: Key, labelKey: "sidebar.keys" },
      { to: "/admin/sessions", icon: MessageSquare, labelKey: "sidebar.sessions" },
      { to: "/admin/skills", icon: Brain, labelKey: "sidebar.skills" },
    ],
  },
];

export function Sidebar() {
  const { t, locale, setLocale } = useTranslation();

  return (
    <aside className="fixed inset-y-0 left-0 z-50 flex flex-col w-[200px] bg-sidebar-background border-r border-sidebar-border">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-5">
        <CliverLogo size={26} />
        <span className="font-semibold text-sm text-foreground">CLIver Lab</span>
      </div>

      {/* Navigation sections */}
      <nav className="flex-1 px-3 space-y-4 overflow-y-auto">
        {navSections.map((section) => (
          <div key={section.titleKey}>
            <div className="px-2 mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              {t(section.titleKey)}
            </div>
            <div className="space-y-0.5">
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-2.5 px-2.5 py-[7px] rounded-md text-[13px] transition-colors",
                      isActive
                        ? "bg-accent text-accent-foreground font-medium border-l-2 border-sidebar-primary"
                        : "text-muted-foreground hover:bg-secondary hover:text-foreground",
                    )
                  }
                >
                  <item.icon className="w-4 h-4 shrink-0" />
                  {t(item.labelKey)}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Bottom section */}
      <div className="px-3 pb-3 space-y-0.5">
        <NavLink
          to="/admin/settings"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-2.5 px-2.5 py-[7px] rounded-md text-[13px] transition-colors",
              isActive
                ? "bg-accent text-accent-foreground font-medium border-l-2 border-sidebar-primary"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground",
            )
          }
        >
          <Settings className="w-4 h-4 shrink-0" />
          {t("sidebar.settings")}
        </NavLink>
        <div className="flex items-center gap-1 px-2 pt-2">
          <button
            onClick={() => setLocale(locale === "en" ? "zh" : "en")}
            className="flex items-center justify-center w-7 h-7 rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
          >
            <Languages className="w-3.5 h-3.5" />
          </button>
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}
