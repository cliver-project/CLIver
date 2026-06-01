/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  theme: {
    extend: {
      colors: {
        cliver: {
          bg: "#0f0f23",
          surface: "#1a1a3e",
          accent: "#60a5fa",
          purple: "#a78bfa",
          teal: "#38bdf8",
        },
      },
    },
  },
  plugins: [],
};
