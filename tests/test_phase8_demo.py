"""Phase 8 end-to-end and honesty tests: the demo manifest, its generated outputs, and the API edges.

The synthetic demo runs once per module into a temporary outputs root served by
the API, so nothing is written into data/outputs. The Step 3 smoke tests are in
test_demo_api.py; these add what a judge relies on: the manifest agrees with the
files it points at, units and labels are honest, and failures are clean.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from depthwizard.demo import api as api_module  # noqa: E402
from depthwizard.demo import synthetic as synthetic_module  # noqa: E402
from depthwizard.demo.__main__ import main as demo_main  # noqa: E402
from depthwizard.demo.api import create_app  # noqa: E402
from depthwizard.demo.process import DEFAULT_CHECKPOINT  # noqa: E402
from depthwizard.demo.synthetic import NOT_YET_MEASURED, DemoError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_CHECKPOINT = (REPO_ROOT / DEFAULT_CHECKPOINT).is_file()
METRIC_PRODUCTS = ("shadow_heights", "calibrated_agl", "dsm", "dtm", "ndsm")
NOT_MEASURED = "not yet measured"


def _walk(obj: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every (dotted key, value) pair in a JSON-like object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}{k}", v
            yield from _walk(v, f"{path}{k}.")
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, path)


def _upload(client: TestClient, path: Path, **form: str) -> Any:
    with path.open("rb") as fh:
        return client.post("/process", files={"file": (path.name, fh, "application/octet-stream")}, data=form)


# --------------------------------------------------------------------------- the synthetic demo, once


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("phase8") / "outputs"
    client = TestClient(create_app(root), raise_server_exceptions=False)
    r = client.post("/demo/synthetic", params={"mesh": "false"})
    assert r.status_code == 200, r.text
    return {"client": client, "root": root, "body": r.json(), "out": root / "demo"}


def test_manifest_on_disk_is_the_response(demo: dict[str, Any]) -> None:
    on_disk = json.loads((demo["out"] / "reports" / "demo_report.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in demo["body"].items() if k != "report"}
    assert on_disk == body
    served = demo["client"].get(demo["body"]["report"])
    assert served.status_code == 200 and served.json() == on_disk


def test_generated_outputs_exist_and_are_served(demo: dict[str, Any]) -> None:
    out, client = demo["out"], demo["client"]
    expected = [
        "input/synthetic_city_a.tif", "input/synthetic_city_a_truth.json",
        "processed/phase4b/synthetic_dsm.tif", "processed/phase4b/phase4b_report.json",
        "processed/phase6/synthetic/phase6_report.json",
        "validation/synthetic/phase5_report.json", "validation/synthetic/error_map_dsm.png",
        "reports/demo_report.json", "reports/demo_report.md",
        "reports/phase2_shadow_heights.json", "reports/phase2_shadow_heights.png",
    ]
    for rel in expected:
        assert (out / rel).is_file(), rel
    png = client.get("/outputs/demo/reports/phase2_shadow_heights.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert client.get("/outputs/demo/reports/demo_report.md").status_code == 200
    assert not (out / "mesh").exists() or not any((out / "mesh").iterdir())  # mesh was switched off


def test_every_recorded_file_is_inside_the_served_root(demo: dict[str, Any]) -> None:
    """The demo writes where it was told (here a temp root), never into the repository's data/outputs."""
    root = demo["root"].resolve()
    # Recorded paths (bare names such as input.metadata.file are not paths; /outputs/... are URLs).
    files = [v for _, v in _walk(demo["body"]) if isinstance(v, str) and ("/" in v or "\\" in v)
             and v.lower().endswith((".tif", ".json", ".png", ".md")) and not v.startswith("/outputs/")]
    assert files, "the manifest records no files"
    for f in files:
        p = Path(f) if Path(f).is_absolute() else (Path.cwd() / f)
        assert root in p.resolve().parents, f


def test_phase2_rows_follow_the_physics_and_the_supplied_truth(demo: dict[str, Any]) -> None:
    body = demo["body"]
    p2 = body["phase2_shadow_physics"]
    md = body["input"]["metadata"]
    specified = {b["name"]: b["height_m"] for b in body["input"]["buildings_specified"]}
    tan = math.tan(math.radians(md["sun_elevation_deg"]))
    assert {r["building"] for r in p2["buildings"]} == set(specified)
    for r in p2["buildings"]:
        assert r["shadow_length_m"] == pytest.approx(r["shadow_length_px"] * md["gsd_m"], rel=1e-12)
        assert r["height_m"] == pytest.approx(r["shadow_length_m"] * tan, rel=1e-12)
        assert r["specified_height_m"] == specified[r["building"]]
        assert r["abs_error_m"] == pytest.approx(abs(r["height_m"] - r["specified_height_m"]), abs=1e-12)
    assert p2["mae_m"] == pytest.approx(np.mean([r["abs_error_m"] for r in p2["buildings"]]), abs=1e-12)
    on_disk = json.loads((demo["out"] / "reports" / "phase2_shadow_heights.json").read_text(encoding="utf-8"))
    assert on_disk["buildings"] == p2["buildings"]


def test_phase6_clicks_are_the_phase6_report_values(demo: dict[str, Any]) -> None:
    """The heights the viewer is told to expect come from Phase 6, not from the demo."""
    s6 = demo["body"]["phase6_surfaces"]
    report6 = json.loads(Path(s6["report"]).read_text(encoding="utf-8"))
    assert s6["clicks"] == report6["products"][s6["acceptance_input"]]["clicks"]
    for c in s6["clicks"]:
        assert c["ndsm_m"] == pytest.approx(c["specified_height_m"], abs=1e-3)


def test_units_and_labels_are_honest(demo: dict[str, Any]) -> None:
    body = demo["body"]
    assert body["mode"] == "ABSOLUTE" and body["units"] == "metres"
    assert body["label"].startswith("SYNTHETIC")
    assert "not a Phase 3 prediction" in body["label"]
    assert all(p["units"] == "metres" for p in body["phase5_validation"]["products"])
    assert "SPECIFIED" in body["phase5_validation"]["label"] and "NOT a real-world" in body["phase5_validation"]["label"]
    assert body["phase5_validation"]["real_world_validation"].startswith(NOT_MEASURED)
    assert NOT_MEASURED in json.dumps(body["phase6_surfaces"]["real_world_validation"])
    text = json.dumps(body).lower()
    assert "zero-shot" not in text and "unseen" not in text


def test_nothing_unmeasured_carries_a_number(demo: dict[str, Any]) -> None:
    body = demo["body"]
    assert body["not_yet_measured"] == NOT_YET_MEASURED
    for key, value in _walk(body):
        leaf = key.rsplit(".", 1)[-1].lower()
        if "uncertainty" in leaf or leaf == "fps":
            assert value == NOT_MEASURED, key


def test_markdown_report_keeps_the_labels(demo: dict[str, Any]) -> None:
    md = (demo["out"] / "reports" / "demo_report.md").read_text(encoding="utf-8")
    assert demo["body"]["label"] in md
    assert "Real-world validation: not yet measured" in md
    assert "## Not yet measured" in md
    for key in NOT_YET_MEASURED:
        assert f"- {key}: {NOT_MEASURED}" in md
    assert "- not available: mesh export was switched off" in md


# --------------------------------------------------------------------------- API edges


def test_demo_synthetic_refuses_a_concurrent_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()

    def slow(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        started.set()
        release.wait(30)
        return {"phase": "8"}

    monkeypatch.setattr(synthetic_module, "run_synthetic_demo", slow)
    app = create_app(tmp_path / "outputs")
    first: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: first.update(r=TestClient(app).post("/demo/synthetic")))
    worker.start()
    try:
        assert started.wait(30)
        second = TestClient(app, raise_server_exceptions=False).post("/demo/synthetic")
        assert second.status_code == 409 and second.json()["error"] == "busy"
    finally:
        release.set()
        worker.join(30)
    assert first["r"].status_code == 200


def test_demo_synthetic_failure_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise DemoError("mesh export failed (exit 1): secret stderr detail")

    monkeypatch.setattr(synthetic_module, "run_synthetic_demo", boom)
    r = TestClient(create_app(tmp_path / "outputs"), raise_server_exceptions=False).post("/demo/synthetic")
    assert r.status_code == 500
    assert r.json() == {"error": "processing_failed", "message": "mesh export failed (exit 1): see the server log"}


def test_oversized_upload_is_refused_and_removed(tmp_path: Path, fixture_tif: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 16)
    root = tmp_path / "outputs"
    r = _upload(TestClient(create_app(root), raise_server_exceptions=False), fixture_tif)
    assert r.status_code == 413 and r.json()["error"] == "too_large"
    uploads = root / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


@pytest.fixture(scope="module")
def absolute_rgb_tif(fixture_tif: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture as a 3-band GeoTIFF with its CRS, transform and sun tags: ABSOLUTE and RGB."""
    import rasterio

    path = tmp_path_factory.mktemp("abs_rgb") / "abs_rgb.tif"
    with rasterio.open(fixture_tif) as src:
        grey, profile, tags = src.read(1), src.profile, src.tags()
    profile.update(count=3)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.stack([grey, np.full_like(grey, 40), grey[:, ::-1]]))
        dst.update_tags(**tags)
    return path


def test_absolute_upload_never_becomes_metres(tmp_path: Path, absolute_rgb_tif: Path) -> None:
    """An ABSOLUTE-capable GeoTIFF upload gets no metric product; its relative field is unitless."""
    client = TestClient(create_app(tmp_path / "outputs", checkpoint=REPO_ROOT / DEFAULT_CHECKPOINT),
                        raise_server_exceptions=False)
    r = _upload(client, absolute_rgb_tif)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "ABSOLUTE" and body["metric_capable_input"] is True
    assert body["units"] != "metres"
    for name in METRIC_PRODUCTS:
        assert body["products"][name]["status"] == "not available"
        assert "SUPPLIED building footprints" in body["products"][name]["reason"]
    # The only metre-valued field is the input's ground sample distance, not a height.
    assert [k for k, v in _walk(body) if k.endswith("_m") and v is not None] == ["metadata.gsd_m"]
    assert body["validation"]["status"] == "not available"
    assert body["metrics"] == {"accuracy": NOT_MEASURED, "uncertainty": NOT_MEASURED}
    rel = body["products"]["relative_height"]
    if HAS_CHECKPOINT:
        assert body["units"] == "unitless" and rel["units"] == "unitless"
        assert "NOT metres" in rel["output_units"] and "NOT metres" in rel["note"]
    else:
        assert body["units"] is None and rel["status"] == "not available"
    saved = client.get(body["outputs"]["report"]).json()
    assert saved["units"] == body["units"] and saved["mode"] == "ABSOLUTE"


# --------------------------------------------------------------------------- CLI


def test_cli_process_writes_the_same_record(tmp_path: Path, png_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "run"
    assert demo_main(["process", str(png_path), "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((out / "process_report.json").read_text(encoding="utf-8"))
    assert printed["mode"] == "RELATIVE" and printed["units"] is None  # single band: nothing produced
    assert printed["products"]["relative_height"]["status"] == "not available"


def test_cli_process_refuses_bad_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(REPO_ROOT)
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    assert demo_main(["process", str(bad), "--out", str(tmp_path / "run")]) == 2
    assert capsys.readouterr().out.startswith("PROCESS FAILED: unsupported file type")
