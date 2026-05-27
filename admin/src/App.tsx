import { Outlet } from "react-router";
import { Sidebar } from "@/components/sidebar";

export function App() {
  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <main className="pl-[200px]">
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
