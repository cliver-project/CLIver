import { NavLink } from "react-router";
import {
  LayoutDashboard,
  Users,
  Workflow,
  ListTodo,
  MessageSquare,
  Sparkles,
  Settings,
  Languages,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ThemeToggle } from "@/components/theme-toggle";
import { CliverLogo } from "@/components/cliver-logo";
import { useTranslation } from "@/i18n";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/admin/dashboard", icon: LayoutDashboard, labelKey: "sidebar.dashboard" },
  { to: "/admin/agents", icon: Users, labelKey: "sidebar.agents" },
  { to: "/admin/workflows", icon: Workflow, labelKey: "sidebar.workflows" },
  { to: "/admin/tasks", icon: ListTodo, labelKey: "sidebar.tasks" },
  { to: "/admin/sessions", icon: MessageSquare, labelKey: "sidebar.sessions" },
  { to: "/admin/skills", icon: Sparkles, labelKey: "sidebar.skills" },
  { to: "/admin/config", icon: Settings, labelKey: "sidebar.config" },
];

export function Sidebar() {
  const { t, locale, setLocale } = useTranslation();

  return (
    <TooltipProvider delayDuration={0}>
      <aside className="fixed inset-y-0 left-0 z-50 flex flex-col items-center w-14 bg-sidebar-background border-r border-sidebar-border py-4 gap-2">
        <NavLink
          to="/admin/dashboard"
          className="flex items-center justify-center w-9 h-9 rounded-lg mb-4"
        >
          <CliverLogo size={28} />
        </NavLink>

        <nav className="flex flex-col items-center gap-1 flex-1">
          {navItems.map((item) => (
            <Tooltip key={item.to}>
              <TooltipTrigger asChild>
                <NavLink
                  to={item.to}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center justify-center w-9 h-9 rounded-md transition-colors",
                      isActive
                        ? "bg-sidebar-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-sidebar-accent hover:text-foreground",
                    )
                  }
                >
                  <item.icon className="w-5 h-5" />
                </NavLink>
              </TooltipTrigger>
              <TooltipContent side="right">{t(item.labelKey)}</TooltipContent>
            </Tooltip>
          ))}
        </nav>

        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => setLocale(locale === "en" ? "zh" : "en")}
              className="flex items-center justify-center w-8 h-8 rounded-md hover:bg-sidebar-accent text-muted-foreground hover:text-foreground transition-colors"
            >
              <Languages className="w-4 h-4" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">
            {locale === "en" ? "中文" : "English"}
          </TooltipContent>
        </Tooltip>
        <ThemeToggle />
      </aside>
    </TooltipProvider>
  );
}
