import { getCollection, type CollectionEntry } from "astro:content";

const base = import.meta.env.BASE_URL;

export type BlogEntry = CollectionEntry<"blog">;

export interface BlogPost {
  slug: string;
  href: string;
  title: string;
  description: string;
  pubDate: Date;
  updatedDate?: Date;
  tags: string[];
  issue: number;
  featured: boolean;
  draft: boolean;
  author: string;
  series: string;
  readingMinutes: number;
  readingTime: string;
}

function countWords(markdown: string): number {
  const text = markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[#>*_~|-]/g, " ");
  return text.split(/\s+/).filter(Boolean).length;
}

function makePost(entry: BlogEntry): BlogPost {
  const { data } = entry;
  const slug = entry.id;
  const readingMinutes = Math.max(1, Math.ceil(countWords(entry.body ?? "") / 220));

  return {
    slug,
    href: `${base}blog/${slug}/`,
    title: data.title,
    description: data.description,
    pubDate: data.pubDate,
    updatedDate: data.updatedDate,
    tags: data.tags,
    issue: data.issue ?? 0,
    featured: data.featured,
    draft: data.draft,
    author: data.author,
    series: data.series,
    readingMinutes,
    readingTime: `${readingMinutes} min read`,
  };
}

export function toBlogPosts(
  entries: BlogEntry[],
  { includeDrafts = import.meta.env.DEV }: { includeDrafts?: boolean } = {},
): BlogPost[] {
  // Issue numbers follow publication order across every entry, so publishing a
  // draft later never renumbers the notes that are already out.
  const issueBySlug = new Map(
    [...entries]
      .sort((a, b) => a.data.pubDate.getTime() - b.data.pubDate.getTime())
      .map((entry, index) => [entry.id, entry.data.issue ?? index + 1]),
  );

  return entries
    .filter((entry) => includeDrafts || !entry.data.draft)
    .map((entry) => makePost(entry))
    .map((post) => ({ ...post, issue: issueBySlug.get(post.slug) ?? post.issue }))
    .sort((a, b) => b.pubDate.getTime() - a.pubDate.getTime());
}

export async function loadBlogPosts(
  options: { includeDrafts?: boolean } = {},
): Promise<BlogPost[]> {
  return toBlogPosts(await getCollection("blog"), options);
}

export function selectFeaturedPost(posts: BlogPost[]): BlogPost | undefined {
  return posts.find((post) => post.featured) ?? posts[0];
}

export function collectTags(posts: BlogPost[]): string[] {
  return Array.from(new Set(posts.flatMap((post) => post.tags))).sort((a, b) =>
    a.localeCompare(b),
  );
}

export function getRelatedPosts(
  post: BlogPost,
  pool: BlogPost[],
  limit = 2,
): BlogPost[] {
  return pool
    .filter((candidate) => candidate.slug !== post.slug)
    .map((candidate) => ({
      post: candidate,
      overlap: candidate.tags.filter((tag) => post.tags.includes(tag)).length,
    }))
    .filter(({ overlap }) => overlap > 0)
    .sort((a, b) => b.overlap - a.overlap || b.post.pubDate.getTime() - a.post.pubDate.getTime())
    .slice(0, limit)
    .map(({ post: candidate }) => candidate);
}

export function formatPostDate(date: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

export function formatPostDateLong(date: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}
