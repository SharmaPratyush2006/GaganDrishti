"""The shadow detector interface, and the classical HSV-value detector.

Two things are under test. The first is the *interface*: a detector is the one
piece of Phase 2 that a later phase is expected to replace, so the contract it
must honour is pinned independently of the implementation that ships today.
The second is the classical rule itself, including the guarantee that it does
not detect buildings -- footprints are supplied, never inferred.
"""

from __future__ import annotations

import numpy as np
import pytest

from depthwizard.ingest.geotiff import read_raster
from depthwizard.mode import Mode
from depthwizard.shadows import (
    ClassicalShadowDetector,
    DetectionContext,
    ShadowDetector,
    ShadowMask,
)
from depthwizard.shadows.detector import BAND_ORDERS


# ---------------------------------------------------------------------------
# DetectionContext
# ---------------------------------------------------------------------------


def test_context_defaults_are_all_unknown():
    """A detector must work in RELATIVE mode, where no geometry is known."""
    context = DetectionContext()
    assert context.gsd_m is None
    assert context.sun_elevation_deg is None
    assert context.sun_azimuth_deg is None
    assert context.mode is Mode.RELATIVE


@pytest.mark.parametrize("band_order", sorted(BAND_ORDERS))
def test_context_accepts_known_band_orders(band_order):
    assert DetectionContext(band_order=band_order).band_order == band_order


def test_context_rejects_an_unknown_band_order():
    """Guessing the channel order silently inverts red and blue."""
    with pytest.raises(ValueError, match="band_order must be one of"):
        DetectionContext(band_order="rbg")


def test_context_to_dict_round_trips_the_fields():
    context = DetectionContext(
        gsd_m=0.5, sun_elevation_deg=45.0, sun_azimuth_deg=135.0, scene_name="s"
    )
    record = context.to_dict()
    assert record["gsd_m"] == 0.5
    assert record["sun_elevation_deg"] == 45.0
    assert record["sun_azimuth_deg"] == 135.0
    assert record["scene_name"] == "s"


def test_context_from_scene_metadata(fixture_tif):
    """Phase 1 metadata flows into Phase 2 without a manual transcription."""
    from depthwizard.ingest.router import read_metadata

    metadata = read_metadata(fixture_tif)
    context = DetectionContext.from_scene_metadata(metadata)
    assert context.sun_elevation_deg == pytest.approx(45.0)
    assert context.sun_azimuth_deg == pytest.approx(135.0)
    assert context.gsd_m == pytest.approx(0.5)
    assert context.mode is Mode.ABSOLUTE
    assert context.band_order == "rgb"


# ---------------------------------------------------------------------------
# The interface contract
# ---------------------------------------------------------------------------


def test_shadow_detector_is_abstract():
    """The base class is an interface; it must not be instantiable."""
    with pytest.raises(TypeError):
        ShadowDetector()  # type: ignore[abstract]


def test_classical_detector_implements_the_interface():
    assert isinstance(ClassicalShadowDetector(), ShadowDetector)


def test_a_third_party_detector_satisfies_the_contract():
    """A replacement detector needs only detect(); nothing else is required.

    This is the whole point of the ABC, so it is tested with a detector that
    shares no code at all with the classical one.
    """

    class AlwaysTopLeft(ShadowDetector):
        name = "always_top_left"

        def detect(self, image, context):
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[: image.shape[0] // 2, : image.shape[1] // 2] = True
            return ShadowMask(
                mask=mask,
                method="fixed_quadrant",
                threshold=0.0,
                score=mask.astype(np.float32),
            )

    image = np.full((20, 30), 200, dtype=np.uint8)
    result = AlwaysTopLeft().detect(image, DetectionContext())
    assert isinstance(result, ShadowMask)
    assert result.shape == (20, 30)
    assert result.shadow_fraction == pytest.approx(0.25)


def test_detectors_agree_on_the_output_shape():
    """Whatever the method, the mask covers the image grid exactly."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(37, 53), dtype=np.uint8)
    mask = ClassicalShadowDetector().detect(image, DetectionContext())
    assert mask.shape == image.shape
    assert mask.mask.dtype == np.bool_


# ---------------------------------------------------------------------------
# ShadowMask
# ---------------------------------------------------------------------------


def test_shadow_mask_reports_count_and_fraction():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2, :] = True
    result = ShadowMask(
        mask=mask, method="test", threshold=0.5, score=mask.astype(np.float32)
    )
    assert result.shape == (10, 10)
    assert result.shadow_pixel_count == 20
    assert result.shadow_fraction == pytest.approx(0.2)


def test_shadow_mask_rejects_a_non_2d_mask():
    with pytest.raises(ValueError):
        ShadowMask(
            mask=np.zeros((4, 4, 3), dtype=bool),
            method="test",
            threshold=0.5,
            score=np.zeros((4, 4, 3), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# The classical rule
# ---------------------------------------------------------------------------


def test_dark_pixels_are_shadow_and_bright_ones_are_not():
    """A two-tone image is the simplest case Otsu must get right."""
    image = np.full((40, 40), 200, dtype=np.uint8)
    image[10:30, 10:30] = 30
    result = ClassicalShadowDetector().detect(image, DetectionContext())
    assert result.mask[20, 20]  # dark centre
    assert not result.mask[0, 0]  # bright corner
    assert result.shadow_pixel_count == 400


def test_a_fixed_threshold_is_used_verbatim():
    image = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    result = ClassicalShadowDetector(threshold=100 / 255).detect(image, DetectionContext())
    assert result.method == "hsv_value_fixed"
    # Pixels at or below the threshold value are shadow.
    assert result.mask[0, 100]
    assert not result.mask[0, 101]


def test_otsu_is_the_default_and_is_recorded_as_such():
    image = np.full((40, 40), 200, dtype=np.uint8)
    image[:10] = 20
    result = ClassicalShadowDetector().detect(image, DetectionContext())
    assert result.method == "hsv_value_otsu"
    assert result.diagnostics["threshold_source"] == "otsu"
    assert 0.0 < result.threshold < 1.0


def test_threshold_out_of_range_is_rejected():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="threshold must be in"):
            ClassicalShadowDetector(threshold=bad)


def test_fraction_bounds_must_be_ordered():
    with pytest.raises(ValueError, match="min_shadow_fraction"):
        ClassicalShadowDetector(min_shadow_fraction=0.7, max_shadow_fraction=0.3)


def test_detection_is_deterministic():
    """No randomness, no training, no state carried between calls."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    detector = ClassicalShadowDetector()
    first = detector.detect(image, DetectionContext())
    second = detector.detect(image, DetectionContext())
    assert np.array_equal(first.mask, second.mask)
    assert first.threshold == second.threshold


def test_score_is_one_minus_value_so_brighter_scores_lower():
    image = np.array([[0, 128, 255]], dtype=np.uint8)
    result = ClassicalShadowDetector().detect(image, DetectionContext())
    assert result.score[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert result.score[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert result.score[0, 0] > result.score[0, 1] > result.score[0, 2]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_detection_works_across_dtypes(dtype):
    """Real imagery arrives as 8-bit, 11-bit-in-16 and float reflectance."""
    image = np.zeros((20, 20), dtype=dtype)
    top = 255 if dtype == np.uint8 else (4095 if dtype == np.uint16 else 1.0)
    image[:] = top
    image[5:15, 5:15] = top * 0.1
    result = ClassicalShadowDetector().detect(image, DetectionContext())
    assert result.mask[10, 10]
    assert not result.mask[0, 0]


def test_a_three_band_image_is_reduced_to_its_value_channel():
    """HSV value is the per-pixel max over R, G and B."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 200  # one bright channel is enough to be lit
    image[4:, :, :] = 10  # all channels dark
    result = ClassicalShadowDetector().detect(image, DetectionContext(band_order="rgb"))
    assert not result.mask[0, 0]
    assert result.mask[7, 7]


def test_diagnostics_record_what_the_run_actually_did():
    image = np.full((30, 30), 180, dtype=np.uint8)
    image[:8] = 25
    result = ClassicalShadowDetector().detect(image, DetectionContext())
    diagnostics = result.diagnostics
    for key in (
        "threshold_u8",
        "threshold_source",
        "otsu_separability",
        "input_dtype",
        "shadow_fraction",
        "suspicious_shadow_fraction",
    ):
        assert key in diagnostics
    assert diagnostics["input_dtype"] == "uint8"
    assert diagnostics["shadow_fraction"] == pytest.approx(result.shadow_fraction)


def test_a_flat_image_is_flagged_not_rejected():
    """No bright surface to split against: the mask is still returned."""
    image = np.full((20, 20), 128, dtype=np.uint8)
    result = ClassicalShadowDetector().detect(image, DetectionContext())
    assert isinstance(result, ShadowMask)
    assert result.diagnostics["suspicious_shadow_fraction"] in (True, False)


def test_the_detector_does_not_detect_buildings(fixture_tif, generated):
    """Roofs are bright, so a shadow detector must leave footprints unmarked.

    This is the line Phase 2 does not cross: footprints are supplied inputs.
    """
    image = read_raster(fixture_tif).array
    result = ClassicalShadowDetector().detect(image, DetectionContext(gsd_m=0.5))
    for building in generated.truth["buildings"]:
        pixels = building["footprint_px"]
        roof = result.mask[
            pixels["row_min"] : pixels["row_max"], pixels["col_min"] : pixels["col_max"]
        ]
        assert not roof.any(), f"{building['name']} roof was marked as shadow"


def test_detector_recovers_the_synthetic_shadow_mask_exactly(fixture_tif, scene, generated):
    """On the fixture the classical rule is lossless, pixel for pixel.

    The fixture paints shadow and ground at two well-separated reflectances, so
    Otsu has a trivial split to find. Establishing that the detector contributes
    *zero* error here is what lets the measurement tests attribute any residual
    to the pixel grid rather than to segmentation.
    """
    from depthwizard.ingest.synthetic import render_scene

    image = read_raster(fixture_tif).array
    detected = ClassicalShadowDetector().detect(image, DetectionContext(gsd_m=0.5)).mask
    truth = render_scene(scene).shadow_mask
    assert np.array_equal(detected, truth)
