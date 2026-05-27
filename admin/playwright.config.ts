import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60000,
  retries: 0,
  use: {
    baseURL: "http://localhost:8321",
    headless: true,
  },
  webServer: {
    command: "echo 'Start the gateway manually: uv run cliver gateway start'",
    port: 8321,
    reuseExistingServer: true,
  },
});
