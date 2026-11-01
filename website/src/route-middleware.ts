import { defineRouteMiddleware } from "@astrojs/starlight/route-data";

const base = import.meta.env.BASE_URL;

export const onRequest = defineRouteMiddleware((context) => {
  const route = context.locals.starlightRoute;
  const slug = route?.entry.slug;

  if (!route || !slug || (slug !== "blog" && !slug.startsWith("blog/"))) {
    return;
  }

  const alreadyLinked = route.head.some(
    ({ tag, attrs }) =>
      tag === "link" &&
      attrs?.rel === "alternate" &&
      attrs?.type === "application/rss+xml",
  );

  if (alreadyLinked) {
    return;
  }

  const site = context.site ?? new URL(context.url.origin);
  const feedUrl = new URL(`${base}blog/rss.xml`, site).href;

  route.head.push({
    tag: "link",
    attrs: {
      rel: "alternate",
      type: "application/rss+xml",
      title: "CLIver Field Notes",
      href: feedUrl,
    },
  });
});
