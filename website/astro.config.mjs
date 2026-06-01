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
          label: "Configuration",
          items: [
            { label: "Overview", slug: "configuration" },
            { label: "Providers & Models", slug: "configuration/providers" },
            { label: "Permissions", slug: "configuration/permissions" },
          ],
        },
        {
          label: "Core Concepts",
          items: [
            { label: "AgentCore", slug: "core-concepts" },
            { label: "Tools", slug: "core-concepts/tools" },
            { label: "Skills", slug: "core-concepts/skills" },
            { label: "Memory & Identity", slug: "core-concepts/memory" },
          ],
        },
        {
          label: "CLI Usage",
          items: [
            { label: "Chat & Commands", slug: "cli" },
            { label: "Sessions", slug: "cli/sessions" },
          ],
        },
        {
          label: "Gateway",
          items: [
            { label: "Overview", slug: "gateway" },
            { label: "Admin Portal", slug: "gateway/admin" },
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
