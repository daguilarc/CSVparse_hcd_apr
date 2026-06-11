# GitHub Pages setup (one-time)

## 1. Enable GitHub Pages

Repo **Settings → Pages → Build and deployment**:

- **Source:** GitHub Actions

## 2. Add secrets (first CI run only, if census caches are not committed)

**Settings → Secrets and variables → Actions:**

| Secret | Purpose |
|--------|---------|
| `IPUMS_API_KEY` | Refresh NHGIS caches when missing |
| `FRED_API_KEY` | Refresh CPI cache when missing |

Skip secrets if you commit processed caches under `docs/data/census/` (see below).

## 3. Optional — commit census caches (recommended)

Copy caches once from `TableA2-models/`:

```bash
mkdir -p docs/data/census
cp TableA2-models/nhgis_cache.json docs/data/census/
cp TableA2-models/nhgis_cache_2018_place_b19013_b01003.json docs/data/census/
cp TableA2-models/cpi_cache.json docs/data/census/
cp TableA2-models/geocode_cache.json docs/data/census/
cp TableA2-models/acs_zcta_income_cache.json docs/data/census/
git add docs/data/census/
git commit -m "Add census cache bundle for Pages CI"
```

Add to `.gitignore` exceptions if `*.json` blocks these files.

## 4. Push to `main`

Workflow `.github/workflows/build-pages.yml` downloads [tablea2.csv](https://data.ca.gov/dataset/81b0841f-2802-403e-b48e-2ef4b751f77c/resource/fe505d9b-8c36-42ba-ba30-08bc4f34e022/download/tablea2.csv), runs the pipeline, and deploys `docs/`.

First full build (with Hierarchical Bayes) may take **hours**.

## 5. Local dry run

```bash
curl -L -o tablea2.csv "$TABLEA2_URL"
python tablea2_parsefilter_repair.py
python scripts/bootstrap_census_caches.py
python scripts/export_pages_catalog.py
# Open docs/index.html via a local static server (fetch requires HTTP):
python -m http.server 8080 --directory docs
```
