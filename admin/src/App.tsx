import { Outlet } from "react-router";
import { Sidebar } from "@/components/sidebar";
import { TopBar } from "@/components/top-bar";

export function App() {
  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <div className="pl-[200px] flex flex-col min-h-screen">
        <TopBar />
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
