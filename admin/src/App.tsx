import { Outlet } from "react-router";
import { Sidebar } from "@/components/sidebar";
import { TopBar } from "@/components/top-bar";
import { useSidebar } from "@/contexts/sidebar-context";

function AppContent() {
  const { collapsed } = useSidebar();

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <div
        id="app-content"
        className="flex flex-col min-h-screen transition-all duration-200"
        style={{ paddingLeft: collapsed ? 52 : 200 }}
      >
        <TopBar />
        <main className="flex-1 p-6 overflow-hidden flex flex-col">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export function App() {
  return <AppContent />;
}
