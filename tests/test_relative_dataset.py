"""The torch Dataset layer: tensor shapes, collation and crop determinism.

Skipped in full without torch. Needs no timm and builds no model -- this layer
only turns numpy samples into tensors.

Runs on the synthetic DFC2019-*shaped* fixtures from ``conftest``. Those are not
DFC2019 data; see the note there.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="Phase 3 dataset layer requires PyTorch")

from depthwizard.relative.config import (  # noqa: E402
    DatasetConfig,
    RelativeConfig,
    SplitConfig,
    TrainingConfig,
)
from depthwizard.relative.data import DatasetError, discover_pairs  # noqa: E402

from conftest import DFC_IMAGE_SUBDIR, write_synthetic_pair  # noqa: E402
from depthwizard.relative.dataset import (  # noqa: E402
    RelativeHeightDataset,
    build_dataloaders,
    build_datasets,
    collate_samples,
    full_length,
    limit_validation,
)


@pytest.fixture
def pairs(dataset_config):
    return discover_pairs(dataset_config)


@pytest.fixture
def relative_config(dataset_config) -> RelativeConfig:
    """A full Phase 3 config sized for a CPU test run."""
    return RelativeConfig(
        dataset=dataset_config,
        training=TrainingConfig(
            image_size=32,
            batch_size=2,
            gradient_accumulation_steps=1,
            epochs=1,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            precision="fp32",
            device="cpu",
        ),
    )


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------


def test_sample_has_the_shapes_the_model_and_loss_expect(pairs, dataset_config):
    dataset = RelativeHeightDataset(pairs, dataset_config, train=False)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["height"].shape == (1, 32, 32)
    assert sample["valid"].shape == (1, 32, 32)
    assert sample["image"].dtype == torch.float32
    assert sample["height"].dtype == torch.float32
    assert sample["valid"].dtype == torch.bool


def test_heights_are_finite_even_where_invalid(pairs, dataset_config):
    dataset = RelativeHeightDataset(pairs, dataset_config, train=False)
    for index in range(min(len(dataset), 9)):
        assert torch.isfinite(dataset[index]["height"]).all()


def test_metadata_survives_as_a_dict(pairs, dataset_config):
    dataset = RelativeHeightDataset(pairs, dataset_config, train=False)
    meta = dataset[0]["metadata"]
    assert meta["split"] == "val"
    assert meta["index"] == 0
    assert isinstance(meta["scene_id"], str)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def test_collate_stacks_tensors_and_keeps_metadata_readable(pairs, dataset_config):
    dataset = RelativeHeightDataset(pairs, dataset_config, train=False)
    batch = collate_samples([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["height"].shape == (2, 1, 32, 32)
    assert batch["valid"].shape == (2, 1, 32, 32)
    assert isinstance(batch["metadata"], list) and len(batch["metadata"]) == 2
    assert isinstance(batch["metadata"][0]["crs"], str)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_validation_tiles_are_the_full_grid_in_order(pairs, dataset_config):
    dataset = RelativeHeightDataset(pairs, dataset_config, train=False)
    # 96x96 at tile 32 = 9 windows per scene, 4 scenes.
    assert len(dataset) == 9 * len(pairs)
    offsets = [(dataset[i]["metadata"]["row_off"], dataset[i]["metadata"]["col_off"])
               for i in range(9)]
    assert offsets == sorted(offsets)


def test_validation_sampling_is_identical_across_epochs(pairs, dataset_config):
    dataset = RelativeHeightDataset(pairs, dataset_config, train=False)
    first = dataset[3]["image"].clone()
    dataset.set_epoch(7)
    assert torch.equal(dataset[3]["image"], first)


def test_training_crops_repeat_within_an_epoch(pairs, dataset_config):
    """Same (seed, epoch, index) must give the same crop, however it is reached."""
    dataset = RelativeHeightDataset(pairs, dataset_config, train=True)
    assert torch.equal(dataset[2]["image"], dataset[2]["image"])


def test_training_crops_change_between_epochs(pairs, dataset_config):
    dataset = RelativeHeightDataset(pairs, dataset_config, train=True)
    windows = set()
    for epoch in range(6):
        dataset.set_epoch(epoch)
        meta = dataset[0]["metadata"]
        windows.add((meta["row_off"], meta["col_off"]))
    # Six epochs drawing from a 65x65 grid of origins: a single repeated window
    # would mean the epoch is not reaching the seed.
    assert len(windows) > 1


def test_persistent_workers_see_every_new_epoch(pairs, dataset_config):
    """Regression: set_epoch() must reach workers that outlive one epoch.

    With ``persistent_workers=True`` the workers keep the dataset copy they
    started with. A plain ``self.epoch = e`` in the main process never reached
    them, so every epoch replayed epoch 0's crops. This uses real worker
    processes -- the configuration the shipped config trains with.
    """
    from torch.utils.data import DataLoader

    dataset = RelativeHeightDataset(pairs, dataset_config, train=True)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        collate_fn=collate_samples,
    )
    reference = RelativeHeightDataset(pairs, dataset_config, train=True)

    seen_windows = set()
    worker_iterators = set()
    for epoch in range(3):
        dataset.set_epoch(epoch)
        reference.set_epoch(epoch)
        metas = [meta for batch in loader for meta in batch["metadata"]]
        worker_iterators.add(id(loader._iterator))

        assert [m["epoch"] for m in metas] == [epoch] * len(dataset)
        # Each worker's crop must be the one the main process computes for the
        # same (seed, epoch, index): the epoch reached the seed, not just the
        # metadata.
        for meta in metas:
            expected = reference[meta["index"]]["metadata"]
            assert (meta["row_off"], meta["col_off"]) == (
                expected["row_off"],
                expected["col_off"],
            )
        seen_windows.add(tuple((m["row_off"], m["col_off"]) for m in metas))

    assert len(worker_iterators) == 1, "workers were not actually persistent"
    assert len(seen_windows) == 3, "epochs replayed the same crops"


def test_two_datasets_with_the_same_seed_agree(pairs, dataset_config):
    a = RelativeHeightDataset(pairs, dataset_config, train=True, epoch=3)
    b = RelativeHeightDataset(pairs, dataset_config, train=True, epoch=3)
    assert torch.equal(a[5]["image"], b[5]["image"])


def test_training_length_is_tiles_per_scene_times_scenes(pairs, dataset_config):
    dataset = RelativeHeightDataset(pairs, dataset_config, train=True)
    assert len(dataset) == len(pairs) * dataset_config.train_tiles_per_scene


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_empty_scene_list_is_refused(dataset_config):
    with pytest.raises(DatasetError, match="zero scenes"):
        RelativeHeightDataset([], dataset_config, train=True)


def test_build_datasets_applies_the_spatial_split(relative_config):
    train, val = build_datasets(relative_config)
    train_scenes = {p.scene_id for p in train.pairs}
    val_scenes = {p.scene_id for p in val.pairs}
    assert train_scenes == {"JAX_004"}
    assert val_scenes == {"OMA_012"}
    assert not train_scenes & val_scenes


def test_build_datasets_returns_no_validation_when_nothing_is_held_out(dataset_config):
    cfg = RelativeConfig(
        dataset=DatasetConfig(
            root=dataset_config.root,
            tile_size=32,
            split=SplitConfig(mode="scene_prefix", val_scene_prefixes=("NOPE",)),
        ),
        training=TrainingConfig(image_size=32, num_workers=0, persistent_workers=False),
    )
    train, val = build_datasets(cfg)
    assert val is None and len(train.pairs) == 4


def test_dataloaders_yield_collated_batches(relative_config):
    train_loader, val_loader = build_dataloaders(relative_config)
    batch = next(iter(train_loader))
    assert batch["image"].shape[0] <= relative_config.training.batch_size
    assert batch["image"].shape[1:] == (3, 32, 32)
    assert val_loader is not None


def test_validation_loader_is_not_shuffled(relative_config):
    _, val_loader = build_dataloaders(relative_config)
    first = [b["metadata"][0]["index"] for b in val_loader]
    second = [b["metadata"][0]["index"] for b in val_loader]
    assert first == second


# ---------------------------------------------------------------------------
# Bounded validation: training.max_val_crops
# ---------------------------------------------------------------------------


@pytest.fixture
def val_dataset(pairs, dataset_config):
    oma = [p for p in pairs if p.city == "OMA"]
    return RelativeHeightDataset(oma, dataset_config, train=False)  # 2 x 9 = 18 crops


@pytest.mark.parametrize("limit", [None, 18, 100])
def test_limit_at_or_above_the_grid_keeps_the_full_dataset(val_dataset, limit):
    assert limit_validation(val_dataset, limit, seed=1) is val_dataset
    assert full_length(val_dataset) == 18


def test_limit_takes_a_fixed_seeded_subset_in_raster_order(val_dataset):
    subset = limit_validation(val_dataset, 5, seed=20260918)
    assert len(subset) == 5
    assert subset.dataset is val_dataset  # the full grid stays reachable
    assert full_length(subset) == 18
    assert list(subset.indices) == sorted(set(subset.indices))
    assert all(0 <= i < 18 for i in subset.indices)
    # Deterministic for a seed, and actually driven by it.
    assert list(limit_validation(val_dataset, 5, seed=20260918).indices) == list(
        subset.indices
    )
    assert any(
        list(limit_validation(val_dataset, 5, seed=s).indices) != list(subset.indices)
        for s in range(10)
    )


def test_subset_samples_keep_their_full_grid_index(val_dataset):
    subset = limit_validation(val_dataset, 4, seed=3)
    assert [subset[i]["metadata"]["index"] for i in range(4)] == list(subset.indices)


def test_limit_spreads_over_every_validation_scene(pairs, dataset_config):
    oma = [p for p in pairs if p.city == "OMA"]
    full = RelativeHeightDataset(oma, dataset_config, train=False)
    subset = limit_validation(full, 12, seed=0)
    assert {subset[i]["metadata"]["stem"] for i in range(12)} == {p.stem for p in oma}


def test_build_dataloaders_applies_max_val_crops(relative_config):
    from dataclasses import replace

    from conftest import captured_depthwizard_logs

    cfg = replace(
        relative_config, training=replace(relative_config.training, max_val_crops=5)
    )
    with captured_depthwizard_logs() as records:
        train_loader, val_loader = build_dataloaders(cfg)
    assert len(val_loader.dataset) == 5
    assert full_length(val_loader.dataset) == 18
    assert len(train_loader.dataset) == 4  # training is untouched
    logged = [r for r in records if r.getMessage() == "validation crops per epoch"]
    assert len(logged) == 1
    assert (logged[0].evaluated, logged[0].available, logged[0].limited) == (5, 18, True)


def test_null_max_val_crops_keeps_the_full_grid(relative_config):
    from dataclasses import replace

    cfg = replace(
        relative_config, training=replace(relative_config.training, max_val_crops=None)
    )
    _, val_loader = build_dataloaders(cfg)
    assert len(val_loader.dataset) == 18


def test_unreadable_raster_fails_at_construction(dataset_config, dfc_like_root):
    """Better to fail now than midway through the first epoch."""
    target = dfc_like_root / DFC_IMAGE_SUBDIR / "JAX_004_001_RGB.tif"
    target.write_bytes(b"not a geotiff")
    pairs = discover_pairs(dataset_config)
    with pytest.raises(DatasetError, match="could not read the header"):
        RelativeHeightDataset(pairs, dataset_config, train=True)


# ---------------------------------------------------------------------------
# The real DFC2019 header shape: no CRS, identity transform, no nodata tag
# ---------------------------------------------------------------------------


def _no_crs_root(tmp_path, stems=("JAX_004_006", "JAX_004_007", "OMA_132_002"), **kwargs):
    root = tmp_path / "DFC2019"
    for stem in stems:
        write_synthetic_pair(root, stem, **{"size": 64, "georeferenced": False, **kwargs})
    return root


def _no_crs_config(root, **overrides):
    return DatasetConfig(
        root=root,
        tile_size=32,
        train_tiles_per_scene=3,
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
        **overrides,
    )


def _no_crs_notices(records):
    return [r for r in records if "without a CRS" in r.getMessage()]


@pytest.mark.parametrize("policy", ["error", "resample_height_to_image"])
def test_no_crs_size_mismatch_is_rejected_at_dataset_construction(tmp_path, policy):
    root = _no_crs_root(tmp_path, stems=("JAX_004_006",), height_size=32)
    cfg = _no_crs_config(root, on_size_mismatch=policy)
    with pytest.raises(DatasetError, match="sizes differ"):
        RelativeHeightDataset(discover_pairs(cfg), cfg, train=True)


def test_no_crs_transform_mismatch_is_rejected_at_dataset_construction(tmp_path):
    import rasterio
    from rasterio.transform import Affine

    root = _no_crs_root(tmp_path, stems=("JAX_004_006",))
    cfg = _no_crs_config(root)
    pair = discover_pairs(cfg)[0]
    with rasterio.open(pair.height_path) as src:
        data, profile = src.read(1), src.profile
    profile["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 3.0)
    with rasterio.open(pair.height_path, "w", **profile) as dst:
        dst.write(data, 1)
    with pytest.raises(DatasetError, match="transforms differ"):
        RelativeHeightDataset([pair], cfg, train=True)


def test_no_crs_notice_is_logged_once_not_per_read(tmp_path):
    from conftest import captured_depthwizard_logs

    root = _no_crs_root(tmp_path)
    cfg = _no_crs_config(root)
    pairs = discover_pairs(cfg)
    with captured_depthwizard_logs() as records:
        dataset = RelativeHeightDataset(pairs, cfg, train=True)
        for epoch in range(2):
            dataset.set_epoch(epoch)
            samples = [dataset[i] for i in range(len(dataset))]
    assert len(samples) == 9  # 3 pairs x 3 crops, each read at least once
    notices = _no_crs_notices(records)
    assert len(notices) == 1
    assert notices[0].not_georeferenced_pairs == 3
    # And nothing else per read mentions georeferencing.
    assert not [r for r in records if "georeferenced" in r.getMessage()]


def test_georeferenced_dataset_logs_no_no_crs_notice(pairs, dataset_config):
    from conftest import captured_depthwizard_logs

    with captured_depthwizard_logs() as records:
        RelativeHeightDataset(pairs, dataset_config, train=False)
    assert _no_crs_notices(records) == []


def test_no_crs_samples_carry_georeferenced_false_through_collate(tmp_path):
    root = _no_crs_root(tmp_path)
    cfg = _no_crs_config(root)
    dataset = RelativeHeightDataset(discover_pairs(cfg), cfg, train=False)
    batch = collate_samples([dataset[0], dataset[1]])
    assert [m["georeferenced"] for m in batch["metadata"]] == [False, False]
    assert [m["crs"] for m in batch["metadata"]] == ["", ""]
    assert batch["valid"].all()


def test_per_city_split_builds_datasets_with_both_cities_on_both_sides(tmp_path):
    stems = [f"{city}_{tile:03d}_{view:03d}" for city in ("JAX", "OMA")
             for tile in (4, 17, 18, 20, 22) for view in (6, 7)]
    root = _no_crs_root(tmp_path, stems=stems, size=32)
    cfg = RelativeConfig(
        dataset=DatasetConfig(
            root=root,
            tile_size=32,
            train_tiles_per_scene=1,
            split=SplitConfig(mode="per_city_scene", val_fraction=0.2),
        ),
        training=TrainingConfig(
            image_size=32, batch_size=2, gradient_accumulation_steps=1, epochs=1,
            num_workers=0, persistent_workers=False, pin_memory=False,
            precision="fp32", device="cpu",
        ),
    )
    train, val = build_datasets(cfg)
    assert val is not None
    assert {p.city for p in train.pairs} == {p.city for p in val.pairs} == {"JAX", "OMA"}
    assert not {p.scene_id for p in train.pairs} & {p.scene_id for p in val.pairs}
    train_stems = {p.stem for p in train.pairs}
    assert ("JAX_004_006" in train_stems) == ("JAX_004_007" in train_stems)
