import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import tailwind from "@astrojs/tailwind";

const site = "https://cliver-project.github.io";
const base = "/CLIver";

export default defineConfig({
  site,
  base,
  trailingSlash: "always",
  integrations: [
    starlight({
      title: "CLIver",
      logo: {
        dark: "./public/icon.png",
        light: "./public/icon.png",
      },
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { label: "Introduction", slug: "getting-started" },
            { label: "Installation", slug: "getting-started/installation" },
          ],
        },
        {
          label: "Core Concepts",
          items: [
            { label: "AgentCore", slug: "core-concepts" },
            { label: "Concurrency & Lifecycle", slug: "core-concepts/concurrency" },
            { label: "Tools", slug: "core-concepts/tools" },
            { label: "Skills", slug: "core-concepts/skills" },
            { label: "System One", slug: "core-concepts/system-one" },
            { label: "Pipelines", slug: "core-concepts/pipelines" },
            { label: "MCP", slug: "core-concepts/mcp" },
            { label: "Permissions & Policy", slug: "core-concepts/policy" },
            { label: "Media Capabilities", slug: "core-concepts/media" },
          ],
        },
        {
          label: "Models",
          items: [
            { label: "Model Bindings", slug: "configuration" },
            { label: "Providers & Presets", slug: "configuration/providers" },
          ],
        },
        {
          label: "CLI Application",
          items: [
            { label: "Overview", slug: "cli" },
          ],
        },
        {
          label: "AgentCore API",
          items: [
            { label: "Python API", slug: "api" },
          ],
        },
      ],
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/cliver-project/CLIver" },
      ],
      customCss: ["./src/styles/custom.css"],
      components: {
        Header: "./src/components/Header.astro",
        PageTitle: "./src/components/PageTitle.astro",
      },
      expressiveCode: {
        themes: ["github-dark", "github-light"],
        styleOverrides: { borderRadius: "0.5rem" },
      },
      pagination: false,
      lastUpdated: true,
      favicon: "/icon.png",
    }),
    tailwind({ applyBaseStyles: false }),
  ],
});
