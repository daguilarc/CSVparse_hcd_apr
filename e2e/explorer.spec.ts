import { expect, test } from "@playwright/test";

test("APR explorer separates Maps and Models controls", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    consoleErrors.push(error.message);
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "California Multifamily Housing APR Explorer" })).toBeVisible();
  await expect(page.locator(".vintage")).toContainText("projects with 5+ dwelling units");
  await expect(page.locator("#status")).toHaveText("");

  await expect(page.locator("#tab-maps")).toHaveClass(/active/);
  await expect(page.locator("#map-geography")).toBeVisible();
  await expect(page.locator("#map-metric")).toBeVisible();
  await expect(page.locator("#models-geo-wrap")).toBeHidden();
  await expect(page.locator(".tab-row #geo")).toHaveCount(1);
  await expect(page.locator("#panel-models #geo")).toHaveCount(0);
  await expect(page.locator("#panel-maps #geo")).toHaveCount(0);
  await expect(page.locator("#map-chart")).toBeVisible();
  await expect(page.locator("#map-unit-hint")).toBeVisible();
  const mapTrace = await page.locator("#map-chart").evaluate((node) => {
    const plot = node as HTMLDivElement & { data?: Array<Record<string, unknown>> };
    const trace = plot.data?.[0] as {
      below?: unknown;
      marker?: { opacity?: unknown; line?: Record<string, unknown> };
    } | undefined;
    return { below: trace?.below, markerOpacity: trace?.marker?.opacity, markerLine: trace?.marker?.line };
  });
  expect(mapTrace).toMatchObject({
    below: "water",
    markerOpacity: 0.92,
    markerLine: {
      color: "rgba(255,255,255,.72)",
      width: 0.45,
    },
  });

  await page.locator("#tab-models").click();
  await expect(page.locator("#tab-models")).toHaveClass(/active/);
  await expect(page.locator("#models-geo-wrap")).toBeVisible();
  await expect(page.locator(".tab-row #geo")).toBeVisible();
  const modelControlOrder = await page.locator("#panel-models .model-grid > label").evaluateAll((labels) =>
    labels.map((label) => label.firstChild?.textContent?.trim() || ""),
  );
  expect(modelControlOrder).toEqual(["Variable (Y)", "Variable (X)", "Model display", "Zero Values"]);
  await expect(page.locator("#model-display")).toBeVisible();
  await expect(page.locator("#zero-values")).toBeVisible();
  const yRange = await page.locator("#model-chart").evaluate((node) => {
    const plot = node as HTMLDivElement & { layout?: { yaxis?: { range?: number[] } } };
    return plot.layout?.yaxis?.range;
  });
  expect(yRange?.[0]).toBe(0);
  await expect(page.locator("label.robustness-below")).toContainText("Robustness Checks");
  await expect(page.locator("#robustness option:checked")).toHaveText("None");
  await expect(page.locator("#build-info")).toContainText("Release 2018-2024");
  await expect(page.locator("#coef-table")).toContainText("Coefficient");
  await expect(page.getByText("Two-part MLE")).toBeVisible();
  await expect(page.getByText("Stationary bootstrap 95% interval")).toBeVisible();
  const legend = await page.locator("#model-chart").evaluate((node) => {
    const plot = node as HTMLDivElement & { layout?: { legend?: Record<string, unknown> } };
    return plot.layout?.legend;
  });
  expect(legend).toMatchObject({
    x: 0.02,
    y: 0.98,
    xanchor: "left",
    yanchor: "top",
  });

  expect(consoleErrors).toEqual([]);
});

test("Models options are catalog neighbors and continuous uses positive-only", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#status")).toHaveText("");
  await page.locator("#tab-models").click();

  const neighborContract = await page.evaluate(async () => {
    const catalog = await fetch("data/releases/2018-2024/catalog.json").then((response) => response.json());
    const geo = (document.querySelector("#geo") as HTMLSelectElement).value;
    const robustness = (document.querySelector("#robustness") as HTMLSelectElement).value;
    const x = (document.querySelector("#x-col") as HTMLSelectElement).value;
    const y = (document.querySelector("#y-col") as HTMLSelectElement).value;
    const xOptions = [...document.querySelectorAll<HTMLOptionElement>("#x-col option")].map((option) => option.value);
    const yOptions = [...document.querySelectorAll<HTMLOptionElement>("#y-col option")].map((option) => option.value);
    return {
      selectedExists: Boolean(catalog[`${geo}:${y}:${x}:${robustness}`]),
      everyXExists: xOptions.every((value) => Boolean(catalog[`${geo}:${y}:${value}:${robustness}`])),
      everyYExists: yOptions.every((value) => Boolean(catalog[`${geo}:${value}:${x}:${robustness}`])),
    };
  });
  expect(neighborContract).toEqual({
    selectedExists: true,
    everyXExists: true,
    everyYExists: true,
  });

  const continuous = await page.evaluate(async () => {
    const catalog = await fetch("data/releases/2018-2024/catalog.json").then((response) => response.json());
    const entry = Object.entries(catalog).find(([, pair]) =>
      (pair as { model_family?: string }).model_family === "continuous",
    );
    if (!entry) {
      throw new Error("No continuous pair in shipped catalog");
    }
    return { key: entry[0], pair: entry[1] };
  });
  const [geo, y, x, robustness] = continuous.key.split(":");
  await page.locator("#geo").selectOption(geo);
  await page.locator("#robustness").selectOption(robustness);
  await page.locator("#y-col").selectOption(y);
  await page.locator("#x-col").selectOption(x);

  const continuousTraces = await page.locator("#model-chart").evaluate((node) => {
    const plot = node as HTMLDivElement & { data?: Array<{ name?: string; y?: number[] }> };
    return {
      mle: plot.data?.find((trace) => trace.name === "MLE")?.y,
      bootstrapUpper: plot.data?.find((trace) => trace.name === "Stationary bootstrap 95% interval")?.y,
    };
  });
  const pair = continuous.pair as {
    views: {
      positive_only: {
        mle: { mean: number[] };
        stationary_bootstrap: { upper: number[] };
      };
    };
  };
  expect(continuousTraces.mle).toEqual(pair.views.positive_only.mle.mean);
  expect(continuousTraces.bootstrapUpper).toEqual(pair.views.positive_only.stationary_bootstrap.upper);
});
