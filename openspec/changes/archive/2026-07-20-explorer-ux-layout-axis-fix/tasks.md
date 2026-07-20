## 1. OpenSpec + chrome

- [x] 1.1 Delta specs for Geography tab-row / Models-only, Models 2×2 order, Maps baseline align, axis floor
- [x] 1.2 Move `#geo` into `.tab-row` after Models; hide on Maps tab
- [x] 1.3 Reorder Models `.model-grid` to Y, X, Model display, Zero Values
- [x] 1.4 Align Maps Geography view / Map metric select baselines (unit hint present)

## 2. Axis framing

- [x] 2.1 Frame axes from observations + mean curves only; if framing y ≥ 0, set y lower bound to 0

## 3. Verification

- [x] 3.1 Update static contracts and e2e for geo placement, 2×2 order, axis floor tokens
- [x] 3.2 Local smoke: Maps aligned selects; Models Geography beside Models; no negative y dead space when obs ≥ 0

## 4. Pairing + axis labels + diagnostics

- [x] 4.1 Ban econ×econ in prune; keep non-identical housing×housing
- [x] 4.2 Housing axis title Dwelling Units (+ per 1,000); econ ticks % / $
- [x] 4.3 Scientific notation for diagnostics with |v| < 1e-5
- [x] 4.4 Prune live release + tests for no econ×econ
