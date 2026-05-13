import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router";
import { QueryClient, QueryClientProvider, QueryCache } from "@tanstack/react-query";
import { I18nProvider } from "@/i18n";
import { App } from "@/App";
import { AuthError } from "@/lib/api";
import LoginPage from "@/pages/login";
import DashboardPage from "@/pages/dashboard";
import NotebooksList from "@/pages/notebooks/list";
import NotebookEditor from "@/pages/notebooks/editor";
import ProjectsList from "@/pages/projects/list";
import ScenariosList from "@/pages/scenarios/list";
import ScenarioDetailPage from "@/pages/scenarios/detail";
import TaskListPage from "@/pages/tasks/list";
import TaskCreatePage from "@/pages/tasks/create";
import TaskDetailPage from "@/pages/tasks/detail";
import KeysList from "@/pages/keys/list";
import AgentListPage from "@/pages/agents/list";
import AgentDetailPage from "@/pages/agents/detail";
import SessionsPage from "@/pages/sessions";
import SessionDetailPage from "@/pages/session-detail";
import SkillsPage from "@/pages/skills";
import SkillDetailPage from "@/pages/skills/detail";
import SkillCreatePage from "@/pages/skills/create";
import ConfigPage from "@/pages/config";
import AdaptersList from "@/pages/adapters/list";
import "./globals.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
  queryCache: new QueryCache({
    onError: (error) => {
      if (error instanceof AuthError) {
        window.location.href = "/admin/login";
      }
    },
  }),
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <I18nProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route path="/admin/login" element={<LoginPage />} />
            <Route path="/admin" element={<App />}>
              <Route index element={<Navigate to="dashboard" replace />} />
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="notebooks" element={<NotebooksList />} />
              <Route path="notebooks/:id" element={<NotebookEditor />} />
              <Route path="projects" element={<ProjectsList />} />
              <Route path="scenarios" element={<ScenariosList />} />
              <Route path="scenarios/:id" element={<ScenarioDetailPage />} />
              <Route path="tasks" element={<TaskListPage />} />
              <Route path="tasks/new" element={<TaskCreatePage />} />
              <Route path="tasks/:name" element={<TaskDetailPage />} />
              <Route path="keys" element={<KeysList />} />
              <Route path="agents" element={<AgentListPage />} />
              <Route path="agents/:name" element={<AgentDetailPage />} />
              <Route path="sessions" element={<SessionsPage />} />
              <Route path="sessions/:source/:id" element={<SessionDetailPage />} />
              <Route path="skills" element={<SkillsPage />} />
              <Route path="skills/new" element={<SkillCreatePage />} />
              <Route path="skills/:name" element={<SkillDetailPage />} />
              <Route path="adapters" element={<AdaptersList />} />
              <Route path="settings" element={<ConfigPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </I18nProvider>
  </StrictMode>,
);
