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
