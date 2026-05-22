import { NavLink } from "react-router";
import {
  LayoutDashboard,
  ListTodo,
  Key,
  MessageSquare,
  Brain,
  Settings,
  Users,
  Plug,
  PanelLeftClose,
  PanelLeft,
  FlaskConical,
  Server,
} from "lucide-react";
import { CliverLogo } from "@/components/cliver-logo";
import { useTranslation } from "@/i18n";
import { useSidebar } from "@/contexts/sidebar-context";
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
      { to: "/admin/chat", icon: MessageSquare, labelKey: "sidebar.chat" },
      { to: "/admin/labs", icon: FlaskConical, labelKey: "sidebar.labs" },
      { to: "/admin/mcp-servers", icon: Server, labelKey: "sidebar.mcpServers" },
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

const linkClasses = ({ isActive }: { isActive: boolean }) =>
  cn(
    "flex items-center gap-2.5 px-2.5 py-[7px] rounded-md text-[13px] transition-colors",
    isActive
      ? "bg-accent text-accent-foreground font-medium border-l-2 border-sidebar-primary"
      : "text-muted-foreground hover:bg-secondary hover:text-foreground",
  );

export function Sidebar() {
  const { t } = useTranslation();
  const { collapsed, toggle } = useSidebar();

  return (
    <aside
      className={cn(
        "fixed inset-y-0 left-0 z-50 flex flex-col bg-sidebar-background border-r border-sidebar-border transition-all duration-200",
        collapsed ? "w-[52px]" : "w-[200px]",
      )}
    >
      {/* Logo */}
      <NavLink
        to="/admin/dashboard"
        className={cn(
          "flex items-center gap-2 px-4 py-3 hover:bg-accent/30 transition-colors border-b border-sidebar-border",
          collapsed && "justify-center px-2",
        )}
      >
        <CliverLogo size={collapsed ? 24 : 28} />
        {!collapsed && (
          <span className="font-semibold text-xs text-foreground">CLIver Lab</span>
        )}
      </NavLink>

      {/* Navigation */}
      <nav className="flex-1 px-2 overflow-y-auto py-2">
        {/* Dashboard */}
        <NavLink to="/admin/dashboard" className={linkClasses} title={t("sidebar.dashboard")}>
          <LayoutDashboard className="w-4 h-4 shrink-0" />
          {!collapsed && t("sidebar.dashboard")}
        </NavLink>

        {navSections.map((section) => (
          <div key={section.titleKey}>
            {!collapsed && (
              <>
                <div className="my-3 mx-1 border-t border-border" />
                <div className="px-2 mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                  {t(section.titleKey)}
                </div>
              </>
            )}
            {collapsed && <div className="my-2 mx-1 border-t border-border" />}
            <div className="space-y-0.5">
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={linkClasses}
                  title={collapsed ? t(item.labelKey) : undefined}
                >
                  <item.icon className="w-4 h-4 shrink-0" />
                  {!collapsed && t(item.labelKey)}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Bottom: Settings + Collapse toggle */}
      <div className="px-2 pb-3 space-y-1">
        <NavLink to="/admin/settings" className={linkClasses} title={t("sidebar.settings")}>
          <Settings className="w-4 h-4 shrink-0" />
          {!collapsed && t("sidebar.settings")}
        </NavLink>

        <button
          type="button"
          onClick={toggle}
          className="flex items-center gap-2.5 px-2.5 py-[7px] rounded-md text-[13px] text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors w-full"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <PanelLeft className="w-4 h-4 shrink-0" />
          ) : (
            <PanelLeftClose className="w-4 h-4 shrink-0" />
          )}
          {!collapsed && "Collapse"}
        </button>
      </div>
    </aside>
  );
}
