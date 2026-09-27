"""Phase 5 error-map raster/figure and the Markdown report generator."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from depthwizard.ingest.geotiff import read_raster
from depthwizard.validation.error_map import ErrorMapError, error_array, plot_error_map, write_error_raster
from depthwizard.validation.metrics import compute_metrics
from depthwizard.validation.report import format_value, render_markdown

GEO_TRANSFORM = Affine(0.5, 0.0, 700000.0, 0.0, -0.5, 3170000.0)


def _read(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            return src.read(1), src.crs, src.transform, src.nodata, (src.height, src.width)


# ---------------------------------------------------------------------------
# Error map
# ---------------------------------------------------------------------------


def test_error_values_and_invalid_pixels():
    p = np.array([[3.0, 1.0], [5.0, 0.0]])
    r = np.array([[1.0, 1.0], [2.0, np.nan]])
    valid = np.isfinite(r)
    e = error_array(p, r, valid)
    assert e[0, 0] == 2.0 and e[0, 1] == 0.0 and e[1, 0] == 3.0
    assert np.isnan(e[1, 1])
    with pytest.raises(ErrorMapError, match="validity mask"):
        error_array(p, r, np.ones_like(valid))


def test_georeferenced_error_raster_preserves_crs_transform_and_nodata(tmp_path):
    rng = np.random.default_rng(3)
    error = rng.normal(size=(300, 280))
    error[:5, :] = np.nan
    record = write_error_raster(tmp_path / "e.tif", error, crs="EPSG:32643", transform=GEO_TRANSFORM, nodata=-9999.0,
                                tags={"UNITS": "metres"})
    data, crs, transform, nodata, shape = _read(tmp_path / "e.tif")
    assert crs.to_string() == "EPSG:32643"
    assert tuple(transform)[:6] == tuple(GEO_TRANSFORM)[:6]
    assert shape == (300, 280) and nodata == -9999.0
    assert (data[:5] == -9999.0).all()
    np.testing.assert_array_equal(data[5:], error[5:].astype(np.float32))
    assert record["ok"] and record["layout"] == "COG"
    assert record["valid_pixels"] == 300 * 280 - 5 * 280


def test_error_raster_without_crs_stays_without_crs(tmp_path):
    error = np.arange(12, dtype=np.float64).reshape(3, 4) - 5.0
    error[0, 0] = np.nan
    transform = Affine(1.0, 0.0, 512.0, 0.0, 1.0, 0.0)  # a pixel-space tile offset, as DFC2019 has
    record = write_error_raster(tmp_path / "e.tif", error, crs=None, transform=transform, nodata=-9999.0)
    data, crs, read_transform, nodata, shape = _read(tmp_path / "e.tif")
    assert crs is None and record["crs"] is None
    assert tuple(read_transform)[:6] == tuple(transform)[:6]
    assert shape == (3, 4) and data[0, 0] == -9999.0 and data[1, 1] == error[1, 1]
    assert record["ok"]


def test_error_raster_refuses_a_nodata_collision(tmp_path):
    with pytest.raises(ErrorMapError, match="collides"):
        write_error_raster(tmp_path / "e.tif", np.array([[-9999.0, 1.0]]), crs=None, transform=None)


def test_error_map_on_fixture_is_prediction_minus_reference(generated, tmp_path):
    truth = read_raster(generated.height_path)
    prediction = truth.array.astype(np.float64) + np.where(truth.array > 0, 1.5, -0.25)
    valid = np.ones(truth.array.shape, dtype=bool)
    error = error_array(prediction, truth.array, valid)
    write_error_raster(tmp_path / "fixture_error.tif", error, crs=truth.crs, transform=truth.transform)
    data, crs, transform, _, shape = _read(tmp_path / "fixture_error.tif")
    assert crs == truth.crs and transform == truth.transform and shape == truth.array.shape
    assert set(np.unique(data).tolist()) == {-0.25, 1.5}


def test_error_figure_is_drawn_and_documents_its_range(tmp_path):
    error = np.linspace(-2.0, 2.0, 400).reshape(20, 20)
    error[0, 0] = 50.0  # a large error must be drawn, not hidden
    error[1, 1] = np.nan
    drawn = plot_error_map(tmp_path / "e.png", error, units="metres", title="test", crs="EPSG:32643",
                           transform=GEO_TRANSFORM, display_percentile=99.0)
    assert (tmp_path / "e.png").is_file()
    assert drawn["max_error"] == 50.0 and drawn["pixels_beyond_display_range"] >= 1
    assert drawn["invalid_pixels"] == 1 and "p99" in drawn["display_range_rule"]
    assert drawn["axes"] == "map coordinates"


def test_error_figure_edge_cases(tmp_path):
    assert plot_error_map(tmp_path / "none.png", np.full((3, 3), np.nan), units="m", title="t") is None
    assert not (tmp_path / "none.png").exists()
    drawn = plot_error_map(tmp_path / "zero.png", np.zeros((3, 3)), units="m", title="t")
    assert drawn["display_range_rule"] == "every error is exactly 0"
    assert drawn["axes"].startswith("pixel indices")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

SECTIONS = ("# Validation Report", "## Dataset", "## Overall Metrics", "## Terrain Metrics", "## Building Metrics",
            "## Confidence Diagnostics", "## Error Map", "## Reproducibility", "## Limitations")


def _product():
    p = np.array([[1.0, 2.0, 4.0, 3.0]])
    r = np.array([[1.0, 1.0, 5.0, 3.0]])
    valid = np.ones_like(p, dtype=bool)
    overall = compute_metrics(p, r, valid, units="metres").to_dict()
    return {
        "product": "agl", "units": "metres",
        "validity": {"total_pixels": 4, "valid_pixels": 4, "excluded_pixels": {}},
        "metrics": {"overall": overall, "terrain": None, "building": compute_metrics(
            p[:, :1], r[:, :1], valid[:, :1], units="metres").to_dict()},
    }


def test_report_contains_every_section_and_the_measured_values():
    result = {
        "validation_kind": "SYNTHETIC", "status": "measured", "label": "specified heights",
        "real_world_validation": "not yet measured",
        "dataset": {"name": "fixture", "training_region": "JAX", "heldout_region": "OMA",
                    "split": {"rationale": "because", "verification": "PASSED"}, "evaluated_tiles": 3},
        "products": [_product()],
    }
    md = render_markdown(result)
    for section in SECTIONS:
        assert section in md
    assert "SYNTHETIC VALIDATION" in md
    assert "| training region | JAX |" in md and "| held-out validation region | OMA |" in md
    assert "| split rationale | because |" in md
    # MAE of errors (0, 1, -1, 0) = 0.5; RMSE = sqrt(0.5).
    assert "| agl | metres | 4 | 0.5000 | 0.7071 |" in md


def test_unavailable_values_render_as_not_yet_measured_with_reasons():
    md = render_markdown({"validation_kind": "SYNTHETIC", "products": [_product()]})
    terrain = md.split("## Terrain Metrics")[1].split("## Building Metrics")[0]
    assert "not yet measured" in terrain and not re.search(r"\d\.\d{4}", terrain)
    building = md.split("## Building Metrics")[1].split("## Confidence")[0]
    assert "insufficient samples" in building  # one pixel: Pearson r is not computed


def test_empty_result_fabricates_nothing():
    md = render_markdown({})
    for section in SECTIONS:
        assert section in md
    assert "not yet measured" in md
    assert not re.search(r"\d+\.\d+", md), "an empty result must not render any number"


def test_refused_run_says_why():
    md = render_markdown({"validation_kind": "REAL-WORLD", "status": "refused",
                          "status_reason": "OMA was used to train", "real_world_validation": "not yet measured"})
    assert "**Run refused:** OMA was used to train" in md
    assert "Real-world validation: **not yet measured**" in md


def test_format_value_never_turns_absence_into_a_number():
    assert format_value(None) == "not yet measured"
    assert format_value(float("nan")) == "not yet measured"
    assert format_value("") == "not yet measured"
    assert format_value(0.0) == "0.0000" and format_value(1234) == "1,234"
    assert format_value(2.5e-6) == "2.500e-06"
