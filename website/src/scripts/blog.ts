// Client-side enhancements for the Field Notes index and post pages.
// Astro does not transform <script lang="ts"> inside a StarlightPage slot,
// so these behaviors live in a module that Vite compiles instead.

const index = document.querySelector<HTMLElement>("[data-blog-index]");

if (index) {
  const filters = Array.from(
    index.querySelectorAll<HTMLButtonElement>("[data-blog-filter]"),
  );
  const cards = Array.from(index.querySelectorAll<HTMLElement>("[data-blog-card]"));
  const count = index.querySelector<HTMLElement>("[data-blog-count]");
  const noResults = index.querySelector<HTMLElement>("[data-blog-no-results]");

  const applyFilter = (filter: string) => {
    let visible = 0;

    cards.forEach((card) => {
      const tags = (card.dataset.postTags ?? "").split(" ");
      const matches = filter === "all" || tags.includes(filter);
      card.hidden = !matches;
      if (matches) visible += 1;
    });

    filters.forEach((button) => {
      const active = button.dataset.blogFilter === filter;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    if (count) {
      count.textContent = `${visible} ${visible === 1 ? "note" : "notes"}`;
    }
    if (noResults) {
      noResults.hidden = visible !== 0;
    }
  };

  filters.forEach((button) => {
    button.addEventListener("click", () => {
      applyFilter(button.dataset.blogFilter ?? "all");
    });
  });
}

const progress = document.querySelector<HTMLElement>("[data-blog-progress]");
const article = document.querySelector<HTMLElement>("[data-blog-article]");

if (progress && article) {
  let frame = 0;

  const update = () => {
    frame = 0;
    const rect = article.getBoundingClientRect();
    const distance = rect.height - window.innerHeight;
    const ratio = distance > 0 ? Math.min(1, Math.max(0, -rect.top / distance)) : 0;
    progress.style.transform = `scaleX(${ratio})`;
  };

  const schedule = () => {
    if (!frame) frame = window.requestAnimationFrame(update);
  };

  update();
  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule, { passive: true });
}

document.querySelectorAll<HTMLAnchorElement>(".blog-toc--mobile a").forEach((link) => {
  link.addEventListener("click", () => {
    link.closest("details")?.removeAttribute("open");
  });
});

// ── Comments (Giscus) ──────────────────────────────────────

const comments = document.querySelector<HTMLElement>("[data-blog-comments]");

if (comments) {
  const container = comments.querySelector<HTMLElement>(".giscus");
  const { repo, repoId, category, categoryId } = comments.dataset;

  if (container && repo && repoId && category && categoryId && !comments.dataset.giscusLoaded) {
    comments.dataset.giscusLoaded = "true";

    const currentTheme = () =>
      document.documentElement.dataset.theme === "light" ? "light" : "dark";

    const script = document.createElement("script");
    script.src = "https://giscus.app/client.js";
    script.async = true;
    script.crossOrigin = "anonymous";

    const attributes: Record<string, string> = {
      "data-repo": repo,
      "data-repo-id": repoId,
      "data-category": category,
      "data-category-id": categoryId,
      "data-mapping": "pathname",
      "data-strict": "1",
      "data-reactions-enabled": "1",
      "data-emit-metadata": "0",
      "data-input-position": "top",
      "data-theme": currentTheme(),
      "data-lang": "en",
      "data-loading": "lazy",
    };

    for (const [name, value] of Object.entries(attributes)) {
      script.setAttribute(name, value);
    }

    container.appendChild(script);

    const sendTheme = () => {
      const frame = container.querySelector<HTMLIFrameElement>("iframe.giscus-frame");
      frame?.contentWindow?.postMessage(
        { giscus: { setConfig: { theme: currentTheme() } } },
        "https://giscus.app",
      );
    };

    window.addEventListener("message", (event) => {
      if (event.origin !== "https://giscus.app") return;
      if (!(event.data as { giscus?: unknown } | null)?.giscus) return;
      sendTheme();
    });

    new MutationObserver(sendTheme).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }
}
