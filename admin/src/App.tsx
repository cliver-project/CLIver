import { Outlet } from "react-router";
import { Sidebar } from "@/components/sidebar";

export function App() {
  return (
    <div className="min-h-screen">
      <Sidebar />
      <main className="pl-14">
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
