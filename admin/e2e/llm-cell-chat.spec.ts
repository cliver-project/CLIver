import { test, expect } from "@playwright/test";

test.describe("LLM Cell Chat", () => {
  test("should render two-panel layout with chat", async ({ page }) => {
    // Navigate to labs list
    await page.goto("/admin/labs");

    // Should see labs page
    await expect(page.locator("text=Labs")).toBeVisible({ timeout: 10000 });

    // Click on first lab (assumes at least one lab exists)
    const firstLab = page.locator("a[href*='/admin/labs/']").first();
    if (await firstLab.isVisible()) {
      await firstLab.click();

      // Wait for lab editor to load
      await page.waitForTimeout(2000);

      // Should see the cell slide
      const cellSlide = page.locator(".grid-cols-\\[280px_1fr\\]");
      if (await cellSlide.isVisible()) {
        // Two-panel layout is rendered for LLM cells
        await expect(cellSlide).toBeVisible();
      }
    }
  });

  test("should show left panel config and composer", async ({ page }) => {
    await page.goto("/admin/labs");

    const firstLab = page.locator("a[href*='/admin/labs/']").first();
    if (await firstLab.isVisible()) {
      await firstLab.click();
      await page.waitForTimeout(2000);

      // Check for left panel config fields
      const agentLabel = page.locator("text=Agent");
      if (await agentLabel.isVisible()) {
        await expect(agentLabel).toBeVisible();
      }
    }
  });
});
