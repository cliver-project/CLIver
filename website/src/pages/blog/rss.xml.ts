import rss from "@astrojs/rss";
import type { APIContext } from "astro";
import { loadBlogPosts } from "../../data/blog-posts";

const base = import.meta.env.BASE_URL;

export async function GET(context: APIContext) {
  if (!context.site) {
    throw new Error("The Astro `site` option is required to build the RSS feed.");
  }

  const publishedPosts = await loadBlogPosts({ includeDrafts: false });
  const feedSite = new URL(`${base}blog/`, context.site);

  return rss({
    title: "CLIver Field Notes",
    description:
      "Short, tested notes on building agents that hold up: the CLIver agent loop, tools, MCP, media pipelines, and production edges.",
    site: feedSite,
    items: publishedPosts.map((post) => ({
      title: post.title,
      description: post.description,
      pubDate: post.pubDate,
      link: `${base}blog/${post.slug}/`,
      author: post.author,
      categories: post.tags,
    })),
    customData: "<language>en-us</language>",
  });
}
