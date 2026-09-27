"""Phase 5: the Markdown validation report.

:func:`render_markdown` turns the JSON-shaped result of a validation run
(:mod:`depthwizard.validation.run`) into Markdown. It **formats**; it never
computes, estimates or defaults a number. Any value that is missing from the
result -- or present only as the ``"not yet measured"`` marker -- is printed as
``not yet measured``, with the recorded reason where one exists. There is no
code path from an absent value to a digit.

Result shape (every key optional; absence renders as not yet measured)::

    validation_kind   "SYNTHETIC" | "REAL-WORLD"
    status            "measured" | "refused"
    status_reason     why a run was refused
    label             one-line banner
    real_world_validation   "not yet measured" or a statement
    dataset           {name, description, training_region, heldout_region,
                       split: {...}, evaluated_tiles, evaluated_images,
                       reference, prediction}
    categories        {source, definitions: {overall, terrain, building}}
    products          [{product, units, description, validity: {...},
                        metrics: {overall|terrain|building: RegionMetrics},
                        metrics_by_confidence: {STATE: RegionMetrics},
                        confidence: {...}, error_map: {...}}]
    reproducibility   {command, config_path, git_commit, inputs, checkpoint,
                       outputs, config}
    limitations       [str, ...]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from depthwizard.validation.confidence import NOT_AVAILABLE
from depthwizard.validation.evaluation import NOT_MEASURED
from depthwizard.validation.masks import BUILDING, OVERALL, TERRAIN

__all__ = ["render_markdown", "write_markdown_report", "format_value"]

_CATEGORY_TITLES = ((OVERALL, "Overall Metrics"), (TERRAIN, "Terrain Metrics"), (BUILDING, "Building Metrics"))
_METRIC_COLUMNS = (("mae", "MAE"), ("rmse", "RMSE"), ("bias", "bias"), ("pearson_r", "Pearson r"))
_DELTA_LABELS = {"delta<1.25": "δ < 1.25", "delta<1.25^2": "δ < 1.25²", "delta<1.25^3": "δ < 1.25³"}
_CONFIDENCE_STATES = ("HIGH", "NOT_ASSESSED", "REDUCED", "UNSUITABLE")


def format_value(value: Any, *, places: int = 4) -> str:
    """A number for a table, or ``not yet measured`` for anything that is not one."""
    if isinstance(value, bool) or value is None:
        return NOT_MEASURED if value is None else str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return NOT_MEASURED
        if value != 0.0 and abs(value) < 10 ** (-places + 1):
            return f"{value:.3e}"
        return f"{value:.{places}f}"
    if isinstance(value, str):
        return value if value else NOT_MEASURED
    return str(value)


def _get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, Mapping) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


class _Notes:
    """Collects footnotes so tables stay narrow and every absence is explained."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def cell(self, metric: Any, context: str) -> str:
        if not isinstance(metric, Mapping):
            return NOT_MEASURED
        value = metric.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return format_value(float(value))
        reason = metric.get("reason")
        if reason:
            self.lines.append(f"{context}: {reason}")
            return f"{NOT_MEASURED} [{len(self.lines)}]"
        return NOT_MEASURED

    def render(self) -> list[str]:
        if not self.lines:
            return []
        return ["", *(f"[{i}] {line}" for i, line in enumerate(self.lines, 1))]


def _metrics_table(products: Sequence[Mapping[str, Any]], getter, label: str) -> list[str]:
    deltas: list[str] = []
    for product in products:
        for d in (_get(getter(product), "deltas") or []):
            name = d.get("name") if isinstance(d, Mapping) else None
            if name and name not in deltas:
                deltas.append(name)
    if not deltas:
        deltas = list(_DELTA_LABELS)
    header = ["product", "units", "valid px", *(t for _, t in _METRIC_COLUMNS),
              *(_DELTA_LABELS.get(d, d) for d in deltas), "δ domain px"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    notes = _Notes()
    for product in products:
        name = str(product.get("product", "?"))
        region = getter(product)
        if not isinstance(region, Mapping):
            cells = [NOT_MEASURED] * (len(header) - 2)
            lines.append("| " + " | ".join([name, str(product.get("units", NOT_MEASURED)), *cells]) + " |")
            continue
        by_name = {d.get("name"): d for d in region.get("deltas", []) if isinstance(d, Mapping)}
        row = [
            name,
            str(region.get("units", NOT_MEASURED)),
            format_value(region.get("valid_pixels")),
            *(notes.cell(region.get(key), f"{name} {label} {title}") for key, title in _METRIC_COLUMNS),
            *(notes.cell(by_name.get(d), f"{name} {label} {_DELTA_LABELS.get(d, d)}") for d in deltas),
            format_value(region.get("delta_domain_pixels")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines + notes.render()


def _dataset_section(result: Mapping[str, Any]) -> list[str]:
    dataset = result.get("dataset") or {}
    split = dataset.get("split") or {}
    rows = [
        ("dataset", dataset.get("name")),
        ("description", dataset.get("description")),
        ("training region", dataset.get("training_region")),
        ("held-out validation region", dataset.get("heldout_region")),
        ("split rationale", split.get("rationale")),
        ("split verified against the checkpoint", split.get("verification")),
        ("evaluated tiles / samples", dataset.get("evaluated_tiles")),
        ("evaluated images", dataset.get("evaluated_images")),
        ("reference data", dataset.get("reference")),
        ("prediction", dataset.get("prediction")),
    ]
    lines = ["| item | value |", "|---|---|"]
    lines += [f"| {k} | {format_value(v)} |" for k, v in rows]
    for product in result.get("products") or []:
        lines.append(f"| valid pixels ({product.get('product', '?')}) | "
                     f"{format_value(_get(product, 'validity', 'valid_pixels'))} |")
    extra = {k: v for k, v in split.items() if k not in ("rationale", "verification")}
    if extra:
        lines += ["", "Split details (as recorded by the run):", "", "```json",
                  json.dumps(extra, indent=2, default=str), "```"]
    return lines


def _validity_lines(products: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    for product in products:
        validity = product.get("validity") or {}
        excluded = validity.get("excluded_pixels") or {}
        detail = ", ".join(f"{k}: {format_value(v)}" for k, v in excluded.items() if v) or "none"
        lines.append(f"- **{product.get('product', '?')}**: {format_value(validity.get('valid_pixels'))} of "
                     f"{format_value(validity.get('total_pixels'))} pixels evaluated; excluded -- {detail}.")
    return lines


def _confidence_section(products: Sequence[Mapping[str, Any]]) -> list[str]:
    if not products:
        return [f"Confidence diagnostics: {NOT_MEASURED}."]
    first = products[0].get("confidence") or {}
    lines = [f"Kind: {first.get('kind', NOT_MEASURED)}.", ""]
    states = first.get("states") or {}
    if states:
        lines += ["| state | code | meaning |", "|---|---|---|"]
        lines += [f"| {name} | {s.get('code')} | {s.get('meaning')} |" for name, s in states.items()]
        lines += ["", f"Precedence (first applicable wins): {' > '.join(first.get('precedence', []))}.", ""]

    lines += ["| product | check | status | source | flagged valid px | fraction of valid |", "|---|---|---|---|---|---|"]
    for product in products:
        checks = _get(product, "confidence", "checks") or {}
        for name in ("shadow_occlusion", "water", "sun_band"):
            check = checks.get(name) or {}
            status = check.get("status", NOT_MEASURED)
            if name == "sun_band" and check.get("available"):
                status = (f"{check.get('sun_band')} ({format_value(check.get('sun_elevation_deg'), places=2)}° vs "
                          f"band {check.get('band_deg')})")
            elif check.get("reason") and not check.get("available"):
                status = f"{status}: {check['reason']}"
            lines.append(
                f"| {product.get('product', '?')} | {name} | {status} | {check.get('source') or NOT_AVAILABLE} | "
                f"{format_value(check.get('flagged_valid_pixels'))} | "
                f"{format_value(check.get('flagged_fraction_of_valid'))} |"
            )
    lines += ["", "| product | " + " | ".join(_CONFIDENCE_STATES) + " |", "|---|" + "---|" * len(_CONFIDENCE_STATES)]
    for product in products:
        counts = _get(product, "confidence", "state_pixels") or {}
        lines.append(f"| {product.get('product', '?')} | " +
                     " | ".join(format_value(counts.get(s)) for s in _CONFIDENCE_STATES) + " |")

    by_conf = [p for p in products if p.get("metrics_by_confidence")]
    if by_conf:
        lines += ["", "Error by confidence state (does the proxy separate good from bad pixels?):", ""]
        lines += ["| product | state | valid px | MAE | RMSE |", "|---|---|---|---|---|"]
        notes = _Notes()
        for product in by_conf:
            name = product.get("product", "?")
            for state in _CONFIDENCE_STATES:
                region = product["metrics_by_confidence"].get(state) or {}
                mae_cell = notes.cell(region.get("mae"), f"{name} {state} MAE")
                rmse_cell = notes.cell(region.get("rmse"), f"{name} {state} RMSE")
                lines.append(f"| {name} | {state} | {format_value(region.get('valid_pixels'))} | {mae_cell} | {rmse_cell} |")
        lines += notes.render()
    return lines


def _error_map_section(products: Sequence[Mapping[str, Any]]) -> list[str]:
    if not products:
        return [f"Error map: {NOT_MEASURED}."]
    lines = []
    for product in products:
        emap = product.get("error_map") or {}
        raster = emap.get("raster") or {}
        figure = emap.get("figure") or {}
        lines += [
            f"**{product.get('product', '?')}** -- {emap.get('description', NOT_MEASURED)}",
            "",
            f"- raster: `{raster.get('path', NOT_MEASURED)}`" + (f" ({emap['rasters_written']} tile rasters in "
                                                                  f"`{emap['raster_dir']}`)" if emap.get("raster_dir") else ""),
            f"- CRS: {raster.get('crs') or 'none (source has none)'}; transform: {raster.get('transform', NOT_MEASURED)}; "
            f"size: {raster.get('width', '?')} x {raster.get('height', '?')}; nodata: {raster.get('nodata', NOT_MEASURED)}",
            f"- read-back check: {'passed' if raster.get('ok') else NOT_MEASURED}"
            + (f" (max |diff| vs memory {format_value(raster.get('max_abs_diff_vs_memory'))})" if raster.get("ok") else ""),
            f"- visualisation: `{figure.get('path', NOT_MEASURED)}`; {figure.get('display_range_rule', NOT_MEASURED)}; "
            f"{format_value(figure.get('pixels_beyond_display_range'))} px beyond the colour range",
            "",
        ]
    return lines


def _reproducibility_section(result: Mapping[str, Any]) -> list[str]:
    repro = result.get("reproducibility") or {}
    lines = [
        f"- command: `{repro.get('command', NOT_MEASURED)}`",
        f"- configuration: `{repro.get('config_path', NOT_MEASURED)}`",
        f"- git commit: {repro.get('git_commit', NOT_MEASURED)}",
        f"- model / checkpoint: {repro.get('checkpoint', NOT_MEASURED)}",
    ]
    for key, value in (repro.get("inputs") or {}).items():
        lines.append(f"- input `{key}`: `{value}`")
    for key, value in (repro.get("outputs") or {}).items():
        lines.append(f"- output `{key}`: `{value}`")
    if repro.get("config"):
        lines += ["", "Resolved configuration:", "", "```json", json.dumps(repro["config"], indent=2, default=str), "```"]
    return lines


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render a validation result as Markdown. Absent values -> not yet measured."""
    kind = result.get("validation_kind") or NOT_MEASURED
    products = [p for p in (result.get("products") or []) if isinstance(p, Mapping)]
    lines = ["# Validation Report", ""]
    lines.append(f"> **{kind} VALIDATION**" + (f" -- {result['label']}" if result.get("label") else ""))
    lines.append(">")
    lines.append(f"> Real-world validation: **{result.get('real_world_validation') or NOT_MEASURED}**")
    if result.get("status") == "refused":
        lines += [">", f"> **Run refused:** {result.get('status_reason') or NOT_MEASURED}"]
    lines.append("")

    lines += ["## Dataset", "", *_dataset_section(result), ""]
    categories = result.get("categories") or {}
    if categories:
        lines.append(f"Categories come from: {categories.get('source', NOT_MEASURED)}.")
        for key, text in (categories.get("definitions") or {}).items():
            lines.append(f"- **{key}**: {text}")
        lines.append("")
    if products:
        lines += ["Valid-pixel accounting (each excluded pixel counted once, under the first reason that applies):",
                  "", *_validity_lines(products), ""]
    lines += ["All metrics use only valid pixels. MAE, RMSE and bias are in the product's units; Pearson r is "
              "dimensionless; δ values are fractions in [0, 1] over the δ domain (valid pixels where prediction and "
              "reference are both > 0). Bias = mean(prediction - reference).", ""]

    for category, title in _CATEGORY_TITLES:
        lines += [f"## {title}", ""]
        if not products:
            lines += [f"Every metric: {NOT_MEASURED}.", ""]
            continue
        lines += [*_metrics_table(products, lambda p, c=category: _get(p, "metrics", c), category), ""]

    lines += ["## Confidence Diagnostics", "", *_confidence_section(products), ""]
    lines += ["## Error Map", "", *_error_map_section(products)]
    lines += ["## Reproducibility", "", *_reproducibility_section(result), ""]
    lines += ["## Limitations", ""]
    limitations = result.get("limitations") or []
    lines += [f"- {item}" for item in limitations] or [f"- {NOT_MEASURED}"]
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(path: str | Path, result: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result), encoding="utf-8")
    return path
