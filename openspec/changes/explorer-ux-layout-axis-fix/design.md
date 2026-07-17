## Context

Live explorer after `explorer-ux-map-model-fixes` still fails chrome and chart framing. Prior decision to move `#geo` into `#panel-models` fixed Maps leakage but broke the intended tab-row placement and 2×2 Models grid.

## Goals / Non-Goals

**Goals:**
- Geography (City/ZIP) in tab row, Models-only visibility.
- Models 2×2: Y | X / Model display | Zero Values.
- Maps select bottoms aligned with unit hint present.
- No y &lt; 0 plot area when observations and mean curves used for framing are all ≥ 0.

**Non-Goals:**
- MF prune, map geometry, opacity, neighbors, continuous positive_only, header copy.

## Decisions

1. **Geography visibility vs placement** — `#geo` returns to `.tab-row` after the tabs; `hidden` (or equivalent) when Maps is active. Rationale: Maps must not show City/ZIP; Models chrome puts Geography beside the Models button.

2. **Axis framing sources** — Range from observation x/y and mean-curve x/y only, not band envelopes. If all framing y ≥ 0, `yaxis.range[0] = 0`.

3. **Maps baseline** — Maps controls grid uses `align-items: end` so unequal label heights still align select bottoms.

## Risks / Trade-offs

- [Bands extend below 0] → Clipped by axis range; interval may look truncated at zero. Acceptable per product rule.
