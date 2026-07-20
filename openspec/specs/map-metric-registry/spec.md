# map-metric-registry Specification

## Purpose
Deterministic archived map-metric registry (`map_metrics.json`) for construction outcomes and ACS deltas.

## Requirements

### Requirement: Enumerate archived mappable construction outcomes

`map_metric_registry.py` SHALL yield one construction map metric for each outcome that is both present as a mappable units column in the prepared city panel and represented by at least one pair in the archived release catalog.

Each entry SHALL include `key`, `y_col`, `metric_col` (`{y_col}_per1000`), `title`, `subtitle`, `cmap_kind: seq`, phase, and applicable geo types.

#### Scenario: Archived BP outcome is mappable

- **WHEN** `df_final` contains `DB_BP_total` and the release catalog contains a pair whose `y_col` is `DB_BP_total`
- **THEN** the registry yields `DB_BP_total_per1000`

#### Scenario: Label exists but panel column does not

- **WHEN** `chart_labels.json` contains an ENT label but `df_final` does not contain the corresponding units column
- **THEN** the registry does not advertise that metric

#### Scenario: Panel column is not archived

- **WHEN** a units column exists but no archived catalog pair uses that outcome
- **THEN** the registry does not advertise it as an archived model outcome

### Requirement: Include ACS delta metrics

The registry SHALL include `population_pct_change` and `income_pct_change` with `cmap_kind: div`, percent units, and explicit applicable geo types.

#### Scenario: ACS deltas present

- **WHEN** the release map registry is built
- **THEN** it includes both ACS delta metrics independently of construction outcome availability

### Requirement: Display titles use structured labels

Construction metric titles SHALL resolve from `docs/chart_labels.json` `outcomes`, keyed by `y_col`. Registry construction SHALL fail when an archived mappable outcome lacks a label.

#### Scenario: Label consistency

- **WHEN** `DB_CO_total` is archived and mappable
- **THEN** its map title matches the Models outcome-control label

### Requirement: Registry output is deterministic

Metric entries SHALL be sorted by a documented stable phase/stream order, and the first entry SHALL be the website/notebook default.

#### Scenario: Repeated release serialization

- **WHEN** the same prepared panel and archived outcomes are serialized twice
- **THEN** `map_metrics.json` ordering is identical

### Requirement: Map metric unit metadata and For-sale titles

Map metric registry entries for `_per1000` metrics SHALL include `unit: "per_1000_pop"`. Map metric titles for owner streams (`mf_owner_*`) SHALL use **For-sale** terminology consistent with `chart_labels.json`.

#### Scenario: Registry encodes per-1k unit

- **WHEN** `build_map_metric_registry` emits `DB_CO_total_per1000`
- **THEN** the registry entry includes `unit: "per_1000_pop"`

#### Scenario: Registry title uses For-sale

- **WHEN** `build_map_metric_registry` emits `mf_owner_CO_total`
- **THEN** the registry entry title uses **For-sale** and does not contain **Owner**
