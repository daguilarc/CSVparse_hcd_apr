## Why

The archived explorer UX change placed Models Geography inside the Models panel grid and left Maps selects and Models chart axis framing wrong. Review showed dead negative y-quadrants, misaligned Map metric vs Geography view selects, and a Models control order that does not match the intended chrome.

## What Changes

- Put Models City/ZIP `#geo` in the shared `.tab-row` to the right of the Models button; show it only when the Models tab is active.
- Models panel controls are exactly a 2×2 grid: Variable (Y) | Variable (X); Model display | Zero Values.
- Maps Geography view and Map metric selects share a common baseline when the unit hint is present.
- Chart axis ranges frame observations + mean curves; when framing y-values are all ≥ 0, the y-axis lower bound is 0 (no negative dead quadrant).

## Capabilities

### Modified Capabilities

- `pages-explorer-ui`: Models Geography placement, Models control order, Maps select baseline alignment, axis framing floor.

## Impact

- [`docs/index.html`](docs/index.html)
- Tests: `tests/test_interactive_map_explorer.py`, `e2e/explorer.spec.ts`
