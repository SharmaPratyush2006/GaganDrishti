"""Phase 6: configuration, raster export round-trip, the height query, and the
SYNTHETIC acceptance run on a fresh Phase 4b export of the Phase 0 fixture.

The fixture's heights were specified, not measured: nothing here is a
real-world accuracy figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import Affine

from depthwizard.config import ConfigError
from depthwizard.surfaces.phase6 import (
    Phase6Error,
    export_surfaces,
    height_at,
    main,
    run_phase6_synthetic,
)
from depthwizard.surfaces.phase6_config import GroundFilterConfig, Phase6Config, Phase6SyntheticConfig, load_phase6_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORM = Affine(0.5, 0.0, 700000.0, 0.0, -0.5, 3170000.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_shipped_config_loads_with_the_documented_parameters():
    cfg = load_phase6_config(REPO_ROOT / "configs" / "phase6.yaml")
    assert cfg.ground.max_building_extent_m is None  # derived from the fixture's known footprints
    assert cfg.ground.max_terrain_slope == 0.05 and cfg.ground.min_object_height_m == 2.0
    assert cfg.synthetic.dsm_products[0] == "synthetic_dsm_known_calibration.tif"


def test_config_is_strict_and_validated():
    with pytest.raises(ConfigError):
        Phase6Config.from_dict({"unknown": 1})
    with pytest.raises(ConfigError):
        Phase6Config.from_dict({"ground": {"window_px": 81}})
    with pytest.raises(ConfigError, match="max_terrain_slope"):
        GroundFilterConfig(max_terrain_slope=-0.1)
    with pytest.raises(ConfigError, match="max_building_extent_m"):
        GroundFilterConfig(max_building_extent_m=0.0)
    with pytest.raises(ConfigError, match="nodata"):
        Phase6Config(nodata=0.0)


# ---------------------------------------------------------------------------
# Export and the height query
# ---------------------------------------------------------------------------


def _exported(tmp_path):
    rows, cols = np.indices((300, 280))
    dsm = 540.0 + 0.01 * cols
    dtm = dsm.copy()
    dsm[100:140, 100:140] += 25.0
    valid = np.ones(dsm.shape, dtype=bool)
    valid[:10, :] = False
    ndsm = dsm - dtm
    ground = ndsm == 0
    records = export_surfaces(tmp_path, dsm=dsm, dtm=dtm, ndsm=ndsm, ground=ground, valid=valid, crs="EPSG:32643",
                              transform=TRANSFORM, nodata=-9999.0, tags={"SYNTHETIC": "true"})
    return records, dsm, dtm, ndsm, valid


def test_export_round_trip_preserves_values_and_metadata(tmp_path):
    records, dsm, dtm, ndsm, valid = _exported(tmp_path)
    for name, expected in (("dsm.tif", dsm), ("dtm.tif", dtm), ("ndsm.tif", ndsm)):
        assert records[name]["ok"] and records[name]["layout"] == "COG"
        with rasterio.open(tmp_path / name) as src:
            assert src.crs.to_string() == "EPSG:32643" and src.transform == TRANSFORM
            assert (src.width, src.height) == (280, 300) and src.dtypes[0] == "float32" and src.nodata == -9999.0
            data = src.read(1)
        assert np.all(data[~valid] == -9999.0)
        np.testing.assert_array_equal(data[valid], expected[valid].astype(np.float32))
    with rasterio.open(tmp_path / "ground_mask.tif") as src:
        mask = src.read(1)
        assert src.dtypes[0] == "uint8" and src.nodata == 255
    assert set(np.unique(mask[valid])) == {0, 1} and np.all(mask[~valid] == 255)


def test_height_at_returns_height_above_ground_not_elevation(tmp_path):
    _exported(tmp_path)
    x, y = 700000.0 + 120.5 * 0.5, 3170000.0 - 120.5 * 0.5  # centre of the raised block
    assert height_at(tmp_path / "ndsm.tif", x, y) == pytest.approx(25.0)
    assert height_at(tmp_path / "dsm.tif", x, y) > 560.0  # the DSM is an absolute elevation
    assert height_at(tmp_path / "ndsm.tif", 690000.0, y) is None  # outside the raster
    assert height_at(tmp_path / "ndsm.tif", x, 3170000.0 - 2.0) is None  # nodata row, never a substitute


# ---------------------------------------------------------------------------
# SYNTHETIC acceptance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase4b(tmp_path_factory) -> Path:
    from depthwizard.surfaces.run import run_synthetic_dsm

    out = tmp_path_factory.mktemp("phase4b")
    run_synthetic_dsm(REPO_ROOT / "configs" / "phase4.yaml", output_dir=out)
    return out / "phase4b_report.json"


def _cfg(report: Path, out: Path) -> Phase6Config:
    return Phase6Config(output_dir=out, synthetic=Phase6SyntheticConfig(
        phase4b_report=report, scene_config=REPO_ROOT / "configs" / "default.yaml"))


@pytest.fixture(scope="module")
def acceptance(phase4b, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase6")
    return run_phase6_synthetic(_cfg(phase4b, out)), out / "synthetic"


def test_acceptance_passes_with_a_tolerance_derived_from_the_measured_dsm_error(acceptance, phase4b):
    report, _ = acceptance
    assert report["acceptance"]["passed"]
    primary = report["products"]["synthetic_dsm_known_calibration.tif"]
    score = primary["score"]
    eps = score["eps_max_abs_dsm_error_m"]
    recorded = json.loads(phase4b.read_text(encoding="utf-8"))["geometric_acceptance"]["max_abs_error_m"]
    assert eps == pytest.approx(recorded, rel=1e-6)  # the same DSM error Phase 4b measured
    assert score["tolerance"]["ndsm_m"] == pytest.approx(2 * eps + score["tolerance"]["float64_rounding_m"])
    assert score["dtm_vs_terrain_truth"]["all_valid_max_abs_m"] <= score["tolerance"]["dtm_m"]
    assert score["ndsm_vs_height_truth"]["all_valid_max_abs_m"] <= score["tolerance"]["ndsm_m"]
    assert score["classification"]["building_px_kept_as_ground"] == 0
    assert score["classification"]["terrain_px_non_ground_after_tin_refinement"] == 0
    assert report["parameters"]["max_building_extent_m"] == 40.0


def test_clicking_each_known_building_returns_its_height_not_its_elevation(acceptance):
    report, _ = acceptance
    clicks = report["products"]["synthetic_dsm_known_calibration.tif"]["clicks"]
    assert {c["building"] for c in clicks} == {"tower_a", "block_b", "slab_c", "low_d"}
    for c in clicks:
        assert c["within_tolerance"]
        assert c["ndsm_m"] == pytest.approx(c["specified_height_m"], abs=2e-4)
        assert c["dsm_m"] > 500.0 and c["dsm_m"] - c["dtm_m"] == pytest.approx(c["ndsm_m"], abs=1e-4)


def test_artifacts_and_labels(acceptance):
    report, out = acceptance
    for product in ("synthetic_dsm_known_calibration", "synthetic_dsm"):
        for name in ("dsm.tif", "dtm.tif", "ndsm.tif", "ground_mask.tif"):
            assert (out / product / name).is_file()
    md = (out / "phase6_report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in md and "not yet measured" in md and "PASSED" in md
    assert report["real_world_validation"].startswith("not yet measured")
    assert report["products"]["synthetic_dsm.tif"]["role"] == "secondary diagnostic"
    assert report["products"]["synthetic_dsm_known_calibration.tif"]["score"]["ndsm"]["negative_px"] == 0


def test_cli_success_and_clear_failure(phase4b, tmp_path, capsys):
    config = tmp_path / "phase6.yaml"
    config.write_text(yaml.safe_dump({"synthetic": {"phase4b_report": str(phase4b),
                                                    "scene_config": str(REPO_ROOT / "configs" / "default.yaml")}}))
    assert main(["synthetic", "--config", str(config), "--output-dir", str(tmp_path / "out")]) == 0
    assert "acceptance: PASSED" in capsys.readouterr().out
    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump({"synthetic": {"phase4b_report": str(tmp_path / "nope.json")}}))
    assert main(["synthetic", "--config", str(missing)]) == 2
    assert "Phase 4b report not found" in capsys.readouterr().out
    with pytest.raises(Phase6Error):
        run_phase6_synthetic(_cfg(tmp_path / "nope.json", tmp_path / "x"))
