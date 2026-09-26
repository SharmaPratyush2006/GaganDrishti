"""Phase 3 config: strictness, validation, and the refusals that matter.

These tests need neither torch nor any dataset. They assert that the config
refuses the things it promises to refuse -- an unknown key, a random encoder
described as pretrained, a split that holds nothing out, a tile size that
disagrees with the model input -- because every one of those, if allowed
through, produces a run whose numbers look fine and mean nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from depthwizard.config import ConfigError
from depthwizard.relative.config import (
    DEFAULT_RELATIVE_CONFIG_PATH,
    CheckpointConfig,
    DatasetConfig,
    InferenceConfig,
    LossConfig,
    ModelConfig,
    RelativeConfig,
    SplitConfig,
    TrainingConfig,
    load_relative_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE3_CONFIG = REPO_ROOT / "configs" / "phase3.yaml"


# ---------------------------------------------------------------------------
# The shipped template
# ---------------------------------------------------------------------------


def test_shipped_template_loads():
    cfg = load_relative_config(PHASE3_CONFIG)
    assert cfg.run_id == "phase3"
    assert cfg.dataset.tile_size == cfg.training.image_size
    assert cfg.model.freeze_encoder is True
    assert cfg.model.pretrained is True
    assert cfg.model.allow_random_init is False


def test_shipped_template_trains_at_256_and_infers_at_512():
    cfg = load_relative_config(PHASE3_CONFIG)
    assert cfg.training.image_size == cfg.dataset.tile_size == 256
    assert cfg.inference.tile_size == 512
    assert cfg.inference.resolved_tile_size(cfg.dataset.tile_size) == 512


def test_inference_tile_size_defaults_to_the_training_tile():
    assert InferenceConfig().resolved_tile_size(256) == 256


@pytest.mark.parametrize("bad", [0, -512, 500])
def test_inference_tile_size_must_be_a_positive_multiple_of_32(bad):
    with pytest.raises(ConfigError, match="inference.tile_size"):
        InferenceConfig(tile_size=bad)


def test_default_config_path_points_at_the_template():
    assert DEFAULT_RELATIVE_CONFIG_PATH == Path("configs/phase3.yaml")
    assert PHASE3_CONFIG.is_file()


def test_template_points_at_a_dataset_that_does_not_exist():
    """The shipped config must not claim a dataset is present.

    DepthWizard ships no DFC2019 data, so the template's root must be absent
    from a fresh clone. If this ever starts passing because someone committed
    imagery, that is the bug.
    """
    cfg = load_relative_config(PHASE3_CONFIG)
    assert not (REPO_ROOT / cfg.dataset.root).exists()


def test_template_split_is_spatially_separated():
    cfg = load_relative_config(PHASE3_CONFIG)
    assert cfg.dataset.split.is_spatially_separated


def test_template_uses_the_per_city_20_percent_split():
    split = load_relative_config(PHASE3_CONFIG).dataset.split
    assert split.mode == "per_city_scene"
    assert split.val_fraction == pytest.approx(0.2)
    assert split.val_scene_prefixes == ()  # the old OMA-only holdout is gone


def test_template_uses_the_verified_nested_layout():
    cfg = load_relative_config(PHASE3_CONFIG)
    assert cfg.dataset.image_subdir == "Training-RGB/Track1-RGB"
    assert cfg.dataset.height_subdir == "Training-Truth/Track1-Truth"
    assert cfg.dataset.image_suffix == "_RGB.tif"
    assert cfg.dataset.height_suffix == "_AGL.tif"


def test_template_is_machine_independent():
    """The shared template names no absolute or machine-specific path."""
    text = PHASE3_CONFIG.read_text(encoding="utf-8")
    cfg = load_relative_config(PHASE3_CONFIG)
    assert not cfg.dataset.root.is_absolute()
    assert not cfg.dataset.root.drive
    for fragment in ("D:", "C:", "DepthWizardData", "\\Users\\", "/home/"):
        assert fragment not in text


# ---------------------------------------------------------------------------
# `extends:` and the git-ignored local config
# ---------------------------------------------------------------------------

LOCAL_CONFIG = REPO_ROOT / "configs" / "phase3.local.yaml"


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_extends_merges_an_override_over_its_base(tmp_path):
    base = _write_yaml(tmp_path / "base.yaml", {
        "run_id": "base",
        "dataset": {"root": "somewhere", "tile_size": 64,
                    "split": {"mode": "explicit", "val_scene_ids": ["OMA_012"]}},
        "training": {"image_size": 64, "num_workers": 0, "persistent_workers": False},
    })
    local = _write_yaml(tmp_path / "local.yaml", {
        "extends": base.name,
        "dataset": {"root": str(tmp_path / "data")},
    })
    cfg = load_relative_config(local)
    assert cfg.dataset.root == tmp_path / "data"  # overridden
    assert cfg.dataset.tile_size == 64  # inherited from a sibling key
    assert cfg.dataset.split.val_scene_ids == ("OMA_012",)  # nested, inherited
    assert cfg.run_id == "base"


def test_extends_keeps_strict_key_checking(tmp_path):
    base = _write_yaml(tmp_path / "base.yaml", _minimal(tmp_path))
    local = _write_yaml(tmp_path / "local.yaml", {
        "extends": base.name, "dataset": {"roots": "typo"},
    })
    with pytest.raises(ConfigError, match="roots"):
        load_relative_config(local)


def test_extends_refuses_a_cycle(tmp_path):
    _write_yaml(tmp_path / "a.yaml", {"extends": "b.yaml"})
    _write_yaml(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="circular"):
        load_relative_config(tmp_path / "a.yaml")


def test_extends_names_a_missing_base(tmp_path):
    local = _write_yaml(tmp_path / "local.yaml", {"extends": "absent.yaml"})
    with pytest.raises(ConfigError, match="not found"):
        load_relative_config(local)


@pytest.mark.skipif(not LOCAL_CONFIG.is_file(), reason="no configs/phase3.local.yaml here")
def test_local_config_only_changes_the_dataset_location():
    template = load_relative_config(PHASE3_CONFIG).to_dict()
    local = load_relative_config(LOCAL_CONFIG).to_dict()
    assert Path(local["dataset"]["root"]).is_absolute()
    assert Path(local["dataset"]["root"]).name == "DFC2019"
    assert local["dataset"]["image_subdir"] == "Training-RGB/Track1-RGB"
    assert local["dataset"]["height_subdir"] == "Training-Truth/Track1-Truth"
    for block in (template, local):
        block["dataset"].pop("root")
    assert local == template  # everything else is inherited unchanged


def _git_check_ignore(*paths: str) -> list[str]:
    import shutil
    import subprocess

    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.split()


def test_dataset_directory_is_git_ignored():
    """Paths need not exist: this checks the ignore rules themselves."""
    paths = [
        "DepthWizardData/DFC2019/Training-RGB/Track1-RGB/JAX_004_006_RGB.tif",
        "DepthWizardData/DFC2019/Training-Truth/Track1-Truth/JAX_004_006_AGL.tif",
    ]
    assert _git_check_ignore(*paths) == paths


def test_local_config_is_git_ignored_but_the_template_is_not():
    assert _git_check_ignore("configs/phase3.local.yaml") == ["configs/phase3.local.yaml"]
    assert _git_check_ignore("configs/phase3.yaml") == []


def test_template_records_relative_units_in_its_dict():
    """to_dict is embedded in every checkpoint; it must round-trip cleanly."""
    cfg = load_relative_config(PHASE3_CONFIG)
    data = cfg.to_dict()
    assert data["dataset"]["root"] == str(cfg.dataset.root)
    assert data["model"]["encoder"] == "dinov2_small"
    # Must be JSON-serialisable: no Path, no tuple, no dataclass left behind.
    import json

    json.loads(json.dumps(data))


# ---------------------------------------------------------------------------
# Strict loading
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "phase3.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _minimal(tmp_path: Path) -> dict:
    return {"dataset": {"root": str(tmp_path / "nowhere"),
                        "split": {"mode": "explicit", "val_scene_ids": ["OMA_012"]}}}


def test_unknown_root_key_is_rejected(tmp_path):
    data = _minimal(tmp_path)
    data["epochs"] = 5  # belongs under `training`
    with pytest.raises(ConfigError, match="epochs"):
        load_relative_config(_write(tmp_path, data))


def test_unknown_nested_key_is_rejected(tmp_path):
    data = _minimal(tmp_path)
    data["training"] = {"batch_sizes": 8}  # typo: plural
    with pytest.raises(ConfigError, match="batch_sizes"):
        load_relative_config(_write(tmp_path, data))


def test_missing_dataset_block_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="dataset"):
        load_relative_config(_write(tmp_path, {"run_id": "x"}))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_relative_config(tmp_path / "absent.yaml")


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_relative_config(path)


# ---------------------------------------------------------------------------
# ModelConfig: the pretrained-vs-random refusal
# ---------------------------------------------------------------------------


def test_random_weights_require_explicit_opt_in():
    with pytest.raises(ConfigError, match="allow_random_init"):
        ModelConfig(pretrained=False)


def test_random_weights_allowed_when_asked_for():
    cfg = ModelConfig(pretrained=False, allow_random_init=True)
    assert cfg.pretrained is False and cfg.allow_random_init is True


def test_unfreezing_the_encoder_is_refused_in_phase_3():
    with pytest.raises(ConfigError, match="freeze_encoder"):
        ModelConfig(freeze_encoder=False)


def test_unknown_encoder_is_refused():
    with pytest.raises(ConfigError, match="model.encoder"):
        ModelConfig(encoder="resnet50")


def test_vit_intermediate_layers_must_have_four_entries():
    with pytest.raises(ConfigError, match="exactly 4"):
        ModelConfig(vit_intermediate_layers=(2, 5, 8))


# ---------------------------------------------------------------------------
# SplitConfig
# ---------------------------------------------------------------------------


def test_scene_prefix_split_needs_prefixes():
    with pytest.raises(ConfigError, match="val_scene_prefixes"):
        SplitConfig(mode="scene_prefix")


def test_explicit_split_needs_ids():
    with pytest.raises(ConfigError, match="val_scene_ids"):
        SplitConfig(mode="explicit")


def test_unknown_split_mode_is_refused():
    with pytest.raises(ConfigError, match="split.mode"):
        SplitConfig(mode="kfold")


@pytest.mark.parametrize(
    "mode,separated",
    [
        ("per_city_scene", True),
        ("scene_prefix", True),
        ("explicit", True),
        ("random_scene", False),
        ("random_tile", False),
    ],
)
def test_spatial_separation_is_reported_honestly(mode, separated):
    kwargs = {"mode": mode}
    if mode == "scene_prefix":
        kwargs["val_scene_prefixes"] = ("OMA",)
    if mode == "explicit":
        kwargs["val_scene_ids"] = ("OMA_012",)
    assert SplitConfig(**kwargs).is_spatially_separated is separated


def test_val_fraction_must_be_a_fraction():
    with pytest.raises(ConfigError, match="val_fraction"):
        SplitConfig(mode="random_scene", val_fraction=1.0)


def test_per_city_split_must_hold_something_out():
    with pytest.raises(ConfigError, match="val_fraction > 0"):
        SplitConfig(mode="per_city_scene", val_fraction=0.0)


def test_per_city_split_needs_a_city_group_in_the_regex(tmp_path):
    with pytest.raises(ConfigError, match=r"\(\?P<city>"):
        DatasetConfig(
            root=tmp_path,
            scene_id_regex=r"^(?P<scene>[A-Za-z]+_\d+)",
            split=SplitConfig(mode="per_city_scene"),
        )


def test_default_regex_separates_tile_view_and_city():
    import re

    match = re.search(DatasetConfig.scene_id_regex, "JAX_004_006")
    assert match.group("scene") == "JAX_004"
    assert match.group("city") == "JAX"


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


def _explicit_split() -> SplitConfig:
    """A valid split, so a DatasetConfig test isolates the field under test.

    Needed because the default ``SplitConfig()`` is itself invalid -- see
    :func:`test_dataset_config_requires_an_explicit_split`.
    """
    return SplitConfig(mode="explicit", val_scene_ids=("OMA_012",))


def test_dataset_config_requires_an_explicit_split(tmp_path):
    """``DatasetConfig(root=...)`` alone does not construct.

    ``split`` defaults to ``SplitConfig()``, whose own default mode is
    ``scene_prefix`` with no prefixes -- which that class refuses. The practical
    effect is that a split must always be chosen explicitly. Recorded as a test
    because it is surprising: the error names ``val_scene_prefixes`` rather than
    saying that ``split`` was never supplied.
    """
    with pytest.raises(ConfigError, match="val_scene_prefixes"):
        DatasetConfig(root=tmp_path)


def test_scene_id_regex_must_name_a_scene_group(tmp_path):
    with pytest.raises(ConfigError, match="scene"):
        DatasetConfig(
            root=tmp_path, scene_id_regex=r"^([A-Z]+_\d+)", split=_explicit_split()
        )


def test_invalid_regex_is_reported_as_such(tmp_path):
    with pytest.raises(ConfigError, match="not a valid regex"):
        DatasetConfig(
            root=tmp_path, scene_id_regex=r"^(?P<scene>[A-Z", split=_explicit_split()
        )


def test_unknown_size_mismatch_policy_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="on_size_mismatch"):
        DatasetConfig(root=tmp_path, on_size_mismatch="stretch", split=_explicit_split())


def test_normalization_stats_must_be_rgb_triples(tmp_path):
    with pytest.raises(ConfigError, match="3 entries"):
        DatasetConfig(root=tmp_path, normalize_mean=(0.5, 0.5), split=_explicit_split())


def test_image_and_height_dirs_derive_from_root(tmp_path):
    cfg = DatasetConfig(
        root=tmp_path, image_subdir="A", height_subdir="B", split=_explicit_split()
    )
    assert cfg.image_dir == tmp_path / "A"
    assert cfg.height_dir == tmp_path / "B"


# ---------------------------------------------------------------------------
# LossConfig
# ---------------------------------------------------------------------------


def test_lambda_si_is_bounded():
    with pytest.raises(ConfigError, match="lambda_si"):
        LossConfig(lambda_si=1.5)


def test_epsilon_must_be_positive():
    with pytest.raises(ConfigError, match="epsilon"):
        LossConfig(epsilon=0.0)


def test_loss_defaults_are_the_documented_ones():
    cfg = LossConfig()
    assert cfg.lambda_si == 0.5
    assert cfg.epsilon == 1.0
    assert cfg.predict_log is True


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


def test_image_size_must_be_a_multiple_of_32():
    with pytest.raises(ConfigError, match="multiple of 32"):
        TrainingConfig(image_size=250)


def test_unknown_precision_is_refused():
    with pytest.raises(ConfigError, match="precision"):
        TrainingConfig(precision="int8")


def test_unknown_device_is_refused():
    with pytest.raises(ConfigError, match="device"):
        TrainingConfig(device="tpu")


def test_persistent_workers_without_workers_is_refused():
    """PyTorch rejects this combination, so the config does too, earlier."""
    with pytest.raises(ConfigError, match="persistent_workers"):
        TrainingConfig(num_workers=0, persistent_workers=True)


def test_effective_batch_size_is_the_product():
    cfg = TrainingConfig(batch_size=8, gradient_accumulation_steps=2)
    assert cfg.effective_batch_size == 16


def test_max_val_crops_defaults_to_a_bounded_subset_and_may_be_null():
    assert TrainingConfig().max_val_crops == 512
    assert TrainingConfig(max_val_crops=None).max_val_crops is None
    assert TrainingConfig(max_val_crops="64").max_val_crops == 64


@pytest.mark.parametrize("bad", [0, -1])
def test_max_val_crops_must_be_positive(bad):
    with pytest.raises(ConfigError, match="max_val_crops"):
        TrainingConfig(max_val_crops=bad)


def test_template_bounds_validation_for_iteration():
    assert load_relative_config(PHASE3_CONFIG).training.max_val_crops == 512


def test_max_steps_per_epoch_may_be_null():
    assert TrainingConfig(max_steps_per_epoch=None).max_steps_per_epoch is None
    assert TrainingConfig(max_steps_per_epoch=3).max_steps_per_epoch == 3


# ---------------------------------------------------------------------------
# RelativeConfig cross-block validation
# ---------------------------------------------------------------------------


def test_tile_size_must_match_image_size(tmp_path):
    with pytest.raises(ConfigError, match="must equal"):
        RelativeConfig(
            dataset=DatasetConfig(
                root=tmp_path,
                tile_size=128,
                split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
            ),
            training=TrainingConfig(image_size=256),
        )


def test_checkpoint_defaults_exclude_the_frozen_encoder():
    assert CheckpointConfig().include_encoder is False
