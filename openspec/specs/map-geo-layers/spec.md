# map-geo-layers Specification

## Purpose
Release `maps.geojson` assembly: geo types, rate denominators, prepared-context construction metrics, and export geometry rules.

## Requirements

### Requirement: Three geo_type values in maps GeoJSON

`assemble_plot_frame()` SHALL emit features tagged `city`, `county_whole`, or `county_residual` in one release GeoJSON.

#### Scenario: All layers present

- **WHEN** map release construction succeeds
- **THEN** at least one feature exists for each of the three geo types

### Requirement: Correct rate denominators per geo_type

City rates SHALL equal jurisdiction units divided by city ACS 2024 population × 1000. County-whole rates SHALL equal all county units divided by county ACS 2024 population × 1000. County-residual rates SHALL equal nonnegative county units minus incorporated-city units divided by county population minus incorporated-city population × 1000.

#### Scenario: City fixture

- **WHEN** a city has 25 units and population 5,000
- **THEN** its rate is 5 per 1,000

#### Scenario: Whole-county fixture

- **WHEN** a county has 1,000 total units and population 200,000
- **THEN** its whole-county rate is 5 per 1,000

#### Scenario: County-residual fixture

- **WHEN** a county has 1,000 units and 200,000 population while incorporated cities total 700 units and 150,000 population
- **THEN** its residual rate is 6 per 1,000

#### Scenario: Residual population guard

- **WHEN** residual population is zero or negative
- **THEN** every residual construction rate for that county is null and not mapped

#### Scenario: Negative residual units

- **WHEN** city rollup units exceed county units because of source mismatch
- **THEN** residual units are clipped to zero before rate calculation and the mismatch is recorded by verification

### Requirement: Construction metrics use prepared release context

Map assembly SHALL use the same `df_final` instance returned by the release's single `prepare_pages_context()` call. It SHALL NOT use the legacy CO-only `load_apr()` path for release construction metrics.

#### Scenario: BP outcome mapped

- **WHEN** `TOTAL_MF_BP_total` is archived and mappable
- **THEN** its city, whole-county, and residual rate properties derive from the prepared release panel

### Requirement: Registry properties exported for applicable layers

`export_maps_geojson()` SHALL include every registry `metric_col` on every geo type named by that entry's applicability metadata, plus `geo_type`, `city_name`, `county_name`, `county_fips`, and stable `feature_id` identity properties.

#### Scenario: Applicable construction property

- **WHEN** a metric applies to all three geography layers
- **THEN** every emitted feature includes its property, using null only where the documented denominator guard applies

### Requirement: Geometry simplification uses meters

Release geometries SHALL be simplified with a 500-meter tolerance while in a projected meter-based CRS and SHALL be converted to EPSG:4326 only after simplification.

#### Scenario: Export order

- **WHEN** `export_maps_geojson()` writes the release artifact
- **THEN** no 500-unit simplify operation is applied to longitude/latitude coordinates

### Requirement: Publication map main remains operable

Refactoring release assembly SHALL preserve the standalone `db_maps.main()` PNG path and its run summary SHALL reference only variables in scope.

#### Scenario: Standalone map smoke test

- **WHEN** `db_maps.main()` runs with required local inputs
- **THEN** it completes without an undefined-variable error and writes its documented PNG outputs
