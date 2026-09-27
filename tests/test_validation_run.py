"""Phase 5 integration: the one-command harness, SYNTHETIC and DFC2019-shaped.

The SYNTHETIC path runs on a fresh Phase 4b export of the Phase 0 fixture and
is cross-checked against the numbers Phase 4b itself recorded.

The DFC path runs on DFC2019-SHAPED synthetic rasters (``conftest``), a fake
checkpoint that carries only a config (it is never used for inference), and an
injected per-tile predictor with a KNOWN error. Nothing here is DFC2019 data
and nothing asserts a real-world accuracy.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.errors import NotGeoreferencedWarning

from depthwizard.validation.config import (
    DfcValidationConfig,
    Phase5Config,
    RegionSplitConfig,
    SyntheticValidationConfig,
    ValidationInputError,
)
from depthwizard.validation.run import main, run_dfc, run_synthetic
from depthwizard.validation.spatial_split import SplitError

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# SYNTHETIC
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase4b(tmp_path_factory) -> Path:
    from depthwizard.surfaces.run import run_synthetic_dsm

    out = tmp_path_factory.mktemp("phase4b")
    run_synthetic_dsm(REPO_ROOT / "configs" / "phase4.yaml", output_dir=out)
    return out / "phase4b_report.json"


def _synthetic_cfg(report: Path, out: Path) -> Phase5Config:
    return Phase5Config(output_dir=out, synthetic=SyntheticValidationConfig(
        phase4b_report=report, scene_config=REPO_ROOT / "configs" / "default.yaml"))


@pytest.fixture(scope="module")
def synthetic_result(phase4b, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase5")
    return run_synthetic(_synthetic_cfg(phase4b, out)), out / "synthetic"


def test_synthetic_pipeline_produces_every_artifact(synthetic_result):
    result, out = synthetic_result
    for name in ("phase5_report.json", "validation_report.md", "error_map_agl.tif", "error_map_agl.png",
                 "error_map_dsm.tif", "error_map_dsm.png", "confidence_agl.tif", "confidence_dsm.tif"):
        assert (out / name).is_file(), name
    assert result["validation_kind"] == "SYNTHETIC" and result["status"] == "measured"
    assert result["real_world_validation"].startswith("not yet measured")
    md = (out / "validation_report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC VALIDATION" in md and "NOT a real-world accuracy claim" in md
    assert json.loads((out / "phase5_report.json").read_text(encoding="utf-8"))["validation_kind"] == "SYNTHETIC"


def test_synthetic_dsm_metrics_match_what_phase4b_recorded(synthetic_result, phase4b):
    result, _ = synthetic_result
    recorded = json.loads(phase4b.read_text(encoding="utf-8"))["calibrations"]["phase4a_synthetic_calibration"]
    dsm = next(p for p in result["products"] if p["product"] == "dsm")
    overall = dsm["metrics"]["overall"]
    assert overall["valid_pixels"] == recorded["dsm_vs_truth"]["valid_pixels"]
    assert overall["mae"]["value"] == pytest.approx(recorded["dsm_vs_truth"]["mae_m"], rel=1e-9)
    assert overall["rmse"]["value"] == pytest.approx(recorded["dsm_vs_truth"]["rmse_m"], rel=1e-9)
    assert overall["units"] == "metres"


def test_synthetic_categories_and_confidence(synthetic_result):
    result, _ = synthetic_result
    agl = next(p for p in result["products"] if p["product"] == "agl")
    terrain, building = agl["metrics"]["terrain"], agl["metrics"]["building"]
    assert terrain["valid_pixels"] + building["valid_pixels"] == agl["validity"]["valid_pixels"]
    assert building["valid_pixels"] == 8300  # the four specified footprints, in pixels
    # Bare-ground AGL is 0: its delta metrics are explained absences, not zeros.
    assert all(d["value"] == "not yet measured" for d in terrain["deltas"])
    checks = agl["confidence"]["checks"]
    assert checks["sun_band"]["sun_band"] == "within_band" and checks["sun_band"]["sun_elevation_deg"] == 45.0
    assert checks["water"]["available"] is False
    assert checks["shadow_occlusion"]["flagged_valid_pixels"] > 0
    assert agl["confidence"]["state_pixels"]["HIGH"] == 0  # water unavailable -> never HIGH


def test_synthetic_error_raster_is_prediction_minus_reference(synthetic_result, phase4b):
    _, out = synthetic_result
    report = json.loads(phase4b.read_text(encoding="utf-8"))
    fixture = Path(report["image_grid"]["source"])
    with rasterio.open(out / "error_map_agl.tif") as e, rasterio.open(report["products"]["synthetic_agl.tif"]) as p, \
            rasterio.open(fixture.with_name(fixture.stem + "_truth_height.tif")) as t:
        assert e.crs == p.crs and e.transform == p.transform and (e.width, e.height) == (p.width, p.height)
        expected = p.read(1).astype(np.float64) - t.read(1).astype(np.float64)
        np.testing.assert_allclose(e.read(1), expected.astype(np.float32), atol=0)


def test_synthetic_cli_exits_zero(phase4b, tmp_path):
    config = tmp_path / "phase5.yaml"
    config.write_text(yaml.safe_dump({"synthetic": {"phase4b_report": str(phase4b),
                                                    "scene_config": str(REPO_ROOT / "configs" / "default.yaml")}}))
    assert main(["synthetic", "--config", str(config), "--output-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "synthetic" / "validation_report.md").is_file()


def test_missing_inputs_fail_clearly(tmp_path, capsys):
    cfg = _synthetic_cfg(tmp_path / "no_such_report.json", tmp_path / "out")
    with pytest.raises(ValidationInputError, match="Phase 4b report not found"):
        run_synthetic(cfg)
    config = tmp_path / "phase5.yaml"
    config.write_text(yaml.safe_dump({"synthetic": {"phase4b_report": str(tmp_path / "missing.json")}}))
    assert main(["synthetic", "--config", str(config)]) == 2
    assert "Phase 4b report not found" in capsys.readouterr().out
    assert main(["dfc", "--config", str(config)]) == 2  # no dfc block: refused, not substituted


# ---------------------------------------------------------------------------
# DFC2019-SHAPED (synthetic rasters)
# ---------------------------------------------------------------------------

TILE = 64
OFFSET_M = 0.5  # the injected predictor's known error


def _write_cls(root: Path, stem: str) -> None:
    from conftest import DFC_HEIGHT_SUBDIR

    cls = np.full((TILE, TILE), 2, dtype=np.uint8)   # ground
    cls[20:30, 20:30] = 6                             # one building, 100 px, off the edge
    cls[40:44, 40:60] = 9                             # water
    cls[50:60, 5:15] = 5                              # vegetation (neither terrain nor building)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(root / DFC_HEIGHT_SUBDIR / f"{stem}_CLS.tif", "w", driver="GTiff", height=TILE,
                           width=TILE, count=1, dtype="uint8") as dst:
            dst.write(cls, 1)


@pytest.fixture
def dfc_setup(tmp_path, synthetic_pair_writer):
    torch = pytest.importorskip("torch", reason="checkpoint configs are read with torch.load")
    from depthwizard.calibration.config import Phase4Config
    from depthwizard.relative.config import DatasetConfig, RelativeConfig, SplitConfig

    root = tmp_path / "dfc"
    stems = ("JAX_004_001", "OMA_012_001")
    for stem in stems:
        synthetic_pair_writer(root, stem, size=TILE, nodata_rows=4, georeferenced=False)
        _write_cls(root, stem)

    def make(split: SplitConfig, epoch: int | None = None, best_epoch: int = 0) -> tuple[Phase5Config, Path]:
        """``epoch`` None -> the final pre-set epoch, as epoch_009.pt of a 10-epoch run."""
        rel = RelativeConfig(dataset=DatasetConfig(root=root, split=split))
        tag = f"{split.mode}_{epoch}"
        rel_path = tmp_path / f"rel_{tag}.yaml"
        rel_path.write_text(yaml.safe_dump(rel.to_dict()))
        ckpt = tmp_path / f"ckpt_{tag}.pt"
        final = rel.training.epochs - 1
        torch.save({"config": rel.to_dict(), "epoch": final if epoch is None else epoch, "best_epoch": best_epoch}, ckpt)
        p4a = tmp_path / f"phase4a_{tag}"
        p4a.mkdir()
        p4 = Phase4Config(checkpoint=ckpt, relative_config=rel_path)
        p4 = Phase4Config.from_dict({**p4.to_dict(), "dataset": {**p4.to_dict()["dataset"], "crop_size": TILE}})
        (p4a / "phase4a_report.json").write_text(json.dumps({"config": p4.to_dict(), "split": {"side": "val"}}))
        records = [
            {"tile_id": "JAX_004_001_r0_c0", "city": "JAX", "image": "JAX_004_001", "scene_id": "JAX_004",
             "row_off": 0, "col_off": 0, "status": "calibrated",
             "fit": {"identifiability": "identifiable_positive", "fit": {"a": 1.0, "b": -1.0}},
             "sun_calibration": {"references": []}},
            {"tile_id": "OMA_012_001_r0_c0", "city": "OMA", "image": "OMA_012_001", "scene_id": "OMA_012",
             "row_off": 0, "col_off": 0, "status": "calibrated",
             "fit": {"identifiability": "statistically_weak", "fit": {"a": 1.0, "b": -1.0 + OFFSET_M}},
             "sun_calibration": {"references": [{"building_id": "OMA_012_001_r0_c0:b0001"}]}},
        ]
        (p4a / "per_tile.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        cfg = Phase5Config(output_dir=tmp_path / f"p5_{tag}", dfc=DfcValidationConfig(
            phase4a_dir=p4a, split=RegionSplitConfig(train_regions=("JAX",), heldout_regions=("OMA",),
                                                     rationale="test")))
        return cfg, rel_path

    return make


def _known_error_predictor(pair, row, col, size):
    """z_rel = log(AGL + 1): with a = 1 the prediction is AGL - 1 - b, i.e. AGL + OFFSET_M here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(pair.height_path) as src:
            agl = src.read(1)[row: row + size, col: col + size].astype(np.float64)
        with rasterio.open(pair.image_path) as src:
            rgb = np.moveaxis(src.read()[:, row: row + size, col: col + size], 0, -1)
    valid = np.isfinite(agl)
    return {"z_rel": np.log(np.where(valid, agl, 0.0) + 1.0), "log_space": True,
            "agl": np.where(valid, agl, np.nan), "agl_valid": valid, "rgb_u8": rgb,
            "transform": [1.0, 0.0, float(col), 0.0, 1.0, float(row)], "crs": None}


def test_dfc_heldout_city_is_evaluated_with_known_error(dfc_setup):
    from depthwizard.relative.config import SplitConfig

    cfg, rel_path = dfc_setup(SplitConfig(mode="scene_prefix", val_scene_prefixes=("OMA",)))
    result = run_dfc(cfg, relative_config=rel_path, tile_predictor=_known_error_predictor)
    assert result["validation_kind"] == "REAL-WORLD" and result["status"] == "measured"
    dataset = result["dataset"]
    assert dataset["training_region"] == "JAX" and dataset["heldout_region"] == "OMA"
    assert dataset["evaluated_tiles"] == 1  # the JAX tile is not held out and is skipped
    assert dataset["phase4a_tiles_skipped"] == {"not in the held-out region": 1}
    assert dataset["split"]["verification"].startswith("PASSED")
    assert dataset["split"]["model_selection"]["status"].startswith("PASSED: final pre-set epoch")

    agl = result["products"][0]
    validity = agl["validity"]
    assert validity["excluded_pixels"]["reference_nonfinite"] == 4 * TILE   # NaN AGL rows
    assert validity["excluded_pixels"]["excluded"] == 100                   # the reference building
    overall = agl["metrics"]["overall"]
    assert overall["mae"]["value"] == pytest.approx(OFFSET_M) and overall["rmse"]["value"] == pytest.approx(OFFSET_M)
    assert overall["units"] == "metres"
    assert agl["metrics"]["building"]["valid_pixels"] == 0  # its only building is the reference
    assert agl["metrics"]["terrain"]["mae"]["value"] == pytest.approx(OFFSET_M)
    checks = agl["confidence"]["checks"]
    assert checks["water"]["flagged_valid_pixels"] == 80
    assert checks["sun_band"]["available"] is False
    assert agl["confidence"]["state_pixels"]["UNSUITABLE"] == 80 and agl["confidence"]["state_pixels"]["HIGH"] == 0

    out = cfg.output_dir / "dfc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(out / "error_maps" / "OMA_012_001_r0_c0.tif") as src:
            assert src.crs is None
            error = src.read(1)
    assert np.allclose(error[error != -9999.0], OFFSET_M, atol=1e-6)
    md = (out / "validation_report.md").read_text(encoding="utf-8")
    assert "REAL-WORLD VALIDATION" in md
    assert "City-held-out model evaluation with LiDAR-anchored per-tile metric calibration." in md
    assert "Training region: JAX" in md and "Held-out region: OMA" in md
    assert "calibration reference buildings are excluded from scoring" in md
    assert "NOT an unseen-city zero-shot result" in md
    assert "if it used tiles of the held-out" not in md


def test_refused_report_does_not_claim_the_measured_framing(dfc_setup):
    from depthwizard.relative.config import SplitConfig

    cfg, rel_path = dfc_setup(SplitConfig(mode="per_city_scene"))
    with pytest.raises(SplitError):
        run_dfc(cfg, relative_config=rel_path, tile_predictor=_known_error_predictor)
    md = (cfg.output_dir / "dfc" / "validation_report.md").read_text(encoding="utf-8")
    assert "LiDAR-anchored per-tile metric calibration" not in md  # nothing was evaluated


def test_dfc_refuses_a_checkpoint_trained_on_the_heldout_city(dfc_setup):
    from depthwizard.relative.config import SplitConfig

    cfg, rel_path = dfc_setup(SplitConfig(mode="per_city_scene"))

    def never_called(*_):  # pragma: no cover - the audit must stop the run first
        raise AssertionError("inference ran despite a failed split audit")

    with pytest.raises(SplitError, match="OMA"):
        run_dfc(cfg, relative_config=rel_path, tile_predictor=never_called)
    md = (cfg.output_dir / "dfc" / "validation_report.md").read_text(encoding="utf-8")
    assert "**Run refused:**" in md and "Real-world validation: **not yet measured**" in md
    assert not (cfg.output_dir / "dfc" / "error_maps").exists()


def test_dfc_refuses_a_best_epoch_checkpoint_selected_on_the_heldout_city(dfc_setup):
    from depthwizard.relative.config import SplitConfig

    # Trained on JAX only, but it is best.pt: epoch 6 of 10, chosen by OMA validation loss.
    cfg, rel_path = dfc_setup(SplitConfig(mode="scene_prefix", val_scene_prefixes=("OMA",)), epoch=6, best_epoch=6)

    def never_called(*_):  # pragma: no cover - the audit must stop the run first
        raise AssertionError("inference ran despite model-selection leakage")

    with pytest.raises(SplitError, match="model-selection leakage"):
        run_dfc(cfg, relative_config=rel_path, tile_predictor=never_called)
    md = (cfg.output_dir / "dfc" / "validation_report.md").read_text(encoding="utf-8")
    assert "**Run refused:** model-selection leakage" in md
    assert "Real-world validation: **not yet measured**" in md
    assert not (cfg.output_dir / "dfc" / "error_maps").exists()


def test_dfc_missing_inputs_fail_clearly(dfc_setup, tmp_path):
    from depthwizard.relative.config import SplitConfig

    cfg, rel_path = dfc_setup(SplitConfig(mode="scene_prefix", val_scene_prefixes=("OMA",)))
    (cfg.dfc.phase4a_dir / "per_tile.jsonl").unlink()
    with pytest.raises(ValidationInputError, match="per_tile.jsonl not found"):
        run_dfc(cfg, relative_config=rel_path, tile_predictor=_known_error_predictor)
    with pytest.raises(ValidationInputError, match="no `dfc` block"):
        run_dfc(Phase5Config(output_dir=tmp_path), tile_predictor=_known_error_predictor)
