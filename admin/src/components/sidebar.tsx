import { NavLink } from "react-router";
import {
  Book,
  FolderOpen,
  LayoutDashboard,
  ListTodo,
  Key,
  MessageSquare,
  Brain,
  Settings,
  Layers,
  Users,
  Plug,
} from "lucide-react";
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
      { to: "/admin/scenarios", icon: Layers, labelKey: "sidebar.scenarios" },
      { to: "/admin/tasks", icon: ListTodo, labelKey: "sidebar.tasks" },
    ],
  },
  {
    titleKey: "sidebar.section.system",
    items: [
      { to: "/admin/keys", icon: Key, labelKey: "sidebar.keys" },
      { to: "/admin/agents", icon: Users, labelKey: "sidebar.agents" },
      { to: "/admin/sessions", icon: MessageSquare, labelKey: "sidebar.sessions" },
      { to: "/admin/skills", icon: Brain, labelKey: "sidebar.skills" },
      { to: "/admin/adapters", icon: Plug, labelKey: "sidebar.adapters" },
    ],
  },
];

export function Sidebar() {
  const { t } = useTranslation();

  return (
    <aside className="fixed inset-y-0 left-0 z-50 flex flex-col w-[200px] bg-sidebar-background border-r border-sidebar-border">
      {/* Logo */}
      <NavLink
        to="/admin/dashboard"
        className="flex flex-col items-center gap-1.5 px-4 py-4 hover:bg-accent/30 transition-colors"
      >
        <CliverLogo size={32} />
        <span className="font-semibold text-xs text-foreground">CLIver Lab</span>
      </NavLink>

      {/* Navigation */}
      <nav className="flex-1 px-3 overflow-y-auto">
        {/* Dashboard */}
        <NavLink
          to="/admin/dashboard"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-2.5 px-2.5 py-[7px] rounded-md text-[13px] transition-colors",
              isActive
                ? "bg-accent text-accent-foreground font-medium border-l-2 border-sidebar-primary"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground",
            )
          }
        >
          <LayoutDashboard className="w-4 h-4 shrink-0" />
          {t("sidebar.dashboard")}
        </NavLink>

        {navSections.map((section) => (
          <div key={section.titleKey}>
            {/* Divider */}
            <div className="my-3 mx-1 border-t border-border" />
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

      {/* Bottom: Settings only */}
      <div className="px-3 pb-3">
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
      </div>
    </aside>
  );
}
